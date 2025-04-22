
from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly
from refined.model_components.config import NER_TAG_TO_IX
from refined.resource_management.aws import S3Manager
from refined.resource_management.resource_manager import ResourceManager
from refined.inference.processor import Refined

data_dir = "/home/morg/dataset/refined/organised_data_dir"

resource_manager = ResourceManager(S3Manager(),
                                    data_dir=data_dir,
                                    entity_set="wikidata",
                                    load_qcode_to_title=True,
                                    load_descriptions_tns=True,
                                    model_name=None,
                                    )

preprocessor = PreprocessorInferenceOnly(
    data_dir=data_dir,
    max_candidates=30,
    transformer_name="roberta-base",
    ner_tag_to_ix=NER_TAG_TO_IX,  # for now include default ner_to_tag_ix can make configurable in future
    entity_set="wikidata",
    use_precomputed_description_embeddings=False
)

refined = Refined(
    model_file_or_model=data_dir+ "/wikipedia_model/model.pt",
    model_config_file_or_model_config=data_dir + "/wikipedia_model/config.json",
    entity_set="wikidata",
    data_dir=data_dir,
    use_precomputed_descriptions = False,
    download_files=False,
    preprocessor=preprocessor,
)

text = "Anarchism is a political philosophy and movement that seeks to abolish all institutions that perpetuate authority, coercion, or hierarchy, primarily targeting the state and capitalism.[1] Anarchism advocates for the replacement of the state with stateless societies and voluntary free associations. A historically left-wing movement, anarchism is usually described as the libertarian wing of the socialist movement (libertarian socialism). Although traces of anarchist ideas are found all throughout history, modern anarchism emerged from the Enlightenment. During the latter half of the 19th and the first decades of the 20th century, the anarchist movement flourished in most parts of the world and had a significant role in workers' struggles for emancipation. Various anarchist schools of thought formed during this period. Anarchists have taken part in several revolutions, most notably in the Paris Commune, the Russian Civil War and the Spanish Civil War, whose end marked the end of the classical era of anarchism. In the last decades of the 20th and into the 21st century, the anarchist movement has been resurgent once more, growing in popularity and influence within anti-capitalist, anti-war and anti-globalisation movements. Anarchists employ diverse approaches, which may be generally divided into revolutionary and evolutionary strategies; there is significant overlap between the two. Evolutionary methods try to simulate what an anarchist society might be like, but revolutionary tactics, which have historically taken a violent turn, aim to overthrow authority and the state. Many facets of human civilization have been influenced by anarchist theory, critique, and praxis."

spans = refined.process_text(text)
print(spans)