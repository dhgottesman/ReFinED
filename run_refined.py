import os
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor
from refined.utilities.general_utils import get_logger
from refined.data_types.base_types import Entity
from dataclasses import asdict, is_dataclass
from typing import Any
from concurrent.futures import wait, FIRST_COMPLETED
import torch
import gc
from threading import Thread
from queue import Queue

logger = get_logger(__name__)

OUTPUT_DIR = "/home/morg/dataset/refined"
refined_model = None

def serialize(obj: Any):
    if is_dataclass(obj):
        return {k: serialize(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, (list, tuple, set)):
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {serialize(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, Entity):  # optional: show Entity custom __repr__ logic
        return {k: v for k, v in vars(obj).items() if v is not None}
    else:
        return obj

def wc(filename: str) -> int:
    """
    Count the number of lines in a file using the Unix `wc -l` command.

    Args:
        filename (str): Path to the file.

    Returns:
        int: Total number of lines in the file.
    """
    return int(subprocess.check_output(['wc', '-l', filename]).split()[0])

def init_worker():
    """
    Worker initializer for `ProcessPoolExecutor`.

    Loads the `Refined` model and its preprocessor into a global variable for reuse across tasks.
    This runs once per worker process.
    """
    print("Starting worker process", flush=True)
    global refined_model
    import os
    from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly
    from refined.model_components.config import NER_TAG_TO_IX
    from refined.inference.processor import Refined

    preprocessor = PreprocessorInferenceOnly(
        data_dir=os.path.join(OUTPUT_DIR, "organised_data_dir"),
        max_candidates=5,
        transformer_name="roberta-base",
        ner_tag_to_ix=NER_TAG_TO_IX,  # for now include default ner_to_tag_ix can make configurable in future
        entity_set="wikidata",
        use_precomputed_description_embeddings=False
    )

    refined_model = Refined(
        model_file_or_model=os.path.join(OUTPUT_DIR, "organised_data_dir", "wikipedia_model", "model.pt"),
        model_config_file_or_model_config=os.path.join(OUTPUT_DIR, "organised_data_dir", "wikipedia_model", "config.json"),
        entity_set="wikidata",
        data_dir=os.path.join(OUTPUT_DIR, "organised_data_dir"),
        use_precomputed_descriptions = False,
        download_files=False,
        preprocessor=preprocessor,
        device="cuda:0",
    )

def process_line(index, line):
    """
    Parses a line of JSON and attaches its input index.

    Args:
        index (int): Line number from the original input file.
        line (str): The raw JSON line from the file.

    Returns:
        dict: Parsed JSON with "line_index" included, or an error object.
    """
    global refined_model
    try:
        obj = json.loads(line)
    except Exception as e:
        obj = {"error": str(e), "raw_line": line}
    obj["line_index"] = index
    return obj

def stream_ndjson(file_path, start_line, end_line, resume_from=-1):
    """
    Generator that yields `(line_index, line)` tuples for a line range in an NDJSON file.

    Args:
        file_path (str): Path to the input .jsonl file.
        start_line (int): Start index of this process's chunk.
        end_line (int): End index of this process's chunk.
        resume_from (int): Skip lines less than or equal to this index.

    Yields:
        tuple[int, str]: Line index and content.
    """
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < start_line or i <= resume_from:
                continue
            if i >= end_line:
                break
            yield (i, line)

def batch_iterator(iterator, batch_size):
    """
    Generator that groups items from an iterator into batches.

    Args:
        iterator (iterable): An iterable yielding items.
        batch_size (int): Maximum number of items per batch.

    Yields:
        list: A list of items in a batch.
    """
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_batch(batch):
    global refined_model
    linked_texts = refined_model.process_text_batch([process_line(idx, line)["text"] for idx, line in batch])
    for i, (index, line) in enumerate(batch):
        batch[i] = process_line(index, line)
        batch[i]["spans"] = serialize(linked_texts[i].spans)
    
    # Free memory
    torch.cuda.empty_cache()
    gc.collect()  # Garbage collection
    
    return batch

# Sentinel to indicate we're done
SENTINEL = (None, None)

def writer_thread(future_queue: Queue, output_path: str):
    print("Starting writer thread", flush=True)
    buffer = {}
    written_batch_index = 0

    with open(output_path, "a") as out_file:
        while True:
            batch_index, result = future_queue.get()
            print(f"batch_index: {batch_index} written_batch_index: {written_batch_index}", flush=True)

            if (batch_index, result) == SENTINEL:
                break  # Exit signal

            if batch_index == written_batch_index:
                for item in sorted(result, key=lambda x: x["line_index"]):
                    out_file.write(json.dumps(item) + "\n")
                written_batch_index += 1

                # Write any buffered results
                while written_batch_index in buffer:
                    result = buffer.pop(written_batch_index)
                    for item in sorted(result, key=lambda x: x["line_index"]):
                        out_file.write(json.dumps(item) + "\n")
                    written_batch_index += 1
            else:
                buffer[batch_index] = result

def parallel_process_ndjson(file_path, output_path, max_workers, batch_size, gpu_id, n_gpus):
    """
    Parallelized entity linking pipeline using a producer-consumer model.
    Handles retries and keeps output in batch order.
    """
    total_lines = 6819853
    lines_per_gpu = total_lines // n_gpus
    start_line = gpu_id * lines_per_gpu
    end_line = (gpu_id + 1) * lines_per_gpu if gpu_id < n_gpus - 1 else total_lines

    logging.info(f"Processing lines {start_line} to {end_line} out of {total_lines} (GPU {gpu_id}/{n_gpus})")
    print(f"Processing lines {start_line} to {end_line} out of {total_lines} (GPU {gpu_id}/{n_gpus})", flush=True)

    input_iterator = batch_iterator(
        stream_ndjson(file_path, start_line, end_line), batch_size
    )

    future_queue = Queue()
    writer = Thread(target=writer_thread, args=(future_queue, output_path))
    writer.start()

    MAX_RETRIES = 10
    retry_counts = {}
    batch_index = 0
    in_flight = {}

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        while True:
            # Fill up the executor with new batches
            while len(in_flight) < max_workers:
                try:
                    batch = next(input_iterator)
                    future = executor.submit(process_batch, batch)
                    in_flight[future] = (batch_index, batch)
                    print(f"Submitted batch {batch_index}", flush=True)
                    batch_index += 1
                except StopIteration:
                    break

            if not in_flight:
                break  # Done processing all batches

            # Wait for any batch to complete
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                b_index, batch = in_flight.pop(future)
                try:
                    result = future.result()
                    print(f"Sending batch {b_index} to writer", flush=True)
                    future_queue.put((b_index, result))
                except Exception as e:
                    retry_counts[b_index] = retry_counts.get(b_index, 0) + 1
                    if retry_counts[b_index] <= MAX_RETRIES:
                        print(f"Retrying batch {b_index} (attempt {retry_counts[b_index]}) due to error: {e}", flush=True)
                        new_future = executor.submit(process_batch, batch)
                        in_flight[new_future] = (b_index, batch)
                    else:
                        print(f"[FAIL] Batch {b_index} failed after {MAX_RETRIES} attempts. Skipping.", flush=True)

    # Signal writer to finish
    future_queue.put(SENTINEL)
    writer.join()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpus", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    file_path = os.path.join(OUTPUT_DIR, "wikipedia_links_aligned_sections_spans.json")
    output_path = os.path.join(OUTPUT_DIR, f"wikipedia_links_sections_{args.gpu_id}.json")
    
    parallel_process_ndjson(
        file_path=file_path,
        output_path=output_path,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        gpu_id=args.gpu_id,
        n_gpus=args.n_gpus
    )