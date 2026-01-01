import argparse
import json
import os
import re
from typing import Set
from urllib.parse import unquote
from random import shuffle

from tqdm import tqdm

from refined.offline_data_generation.preprocessing_utils import DENY_CLASSES
from refined.resource_management.loaders import load_redirects, load_instance_of, load_wikipedia_to_qcode

anchor_tag_pattern = re.compile('<a href="([^"]+)">([^>]+)</a>')
section_pattern = re.compile(r"Section::::(.*?)\n")

def main():
    parser = argparse.ArgumentParser(description='Process cleaned Wikipedia, extract links, merge files.')
    parser.add_argument(
        "--clean_wiki_dir",
        type=str,
        default='output',
        help="Directory where clean Wikipedia is stored (processed by clean_wikipedia_.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='cleaned_output',
        help="Directory where the lookups will be stored"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="mode for testing (only processes first 500 lines)"
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip('/')
    args.clean_wiki_dir = args.clean_wiki_dir.rstrip('/')
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use "
                         f"--overwrite_output_dir to overwrite.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    merge_files_and_extract_links(args.clean_wiki_dir, args.clean_wiki_dir, args.args.output_dir)

def clean_text_with_spans(text):
    hyperlinks = []
    section_starts = []

    output = []
    i = 0  # cursor in original text
    j = 0  # cursor in output text

    while i < len(text):
        # Check for section
        sec_match = section_pattern.match(text, i)
        if sec_match:
            section_label = sec_match.group(1).strip()
            if (len(section_starts) == 0 or section_starts[-1] != j) and len(section_label) > 0:
                # Add section start only if it's not empty and not the same as the previous one
                section_starts.append(j)
            i = sec_match.end()
            continue
            
        # Check for anchor tag
        link_match = anchor_tag_pattern.match(text, i)
        if link_match:
            href, surface = link_match.group(1), link_match.group(2)
            hyperlinks.append({
                "uri": unquote(href).replace(" ", "_"),
                "surface_form": surface,
                "start": j,
                "end": j + len(surface)
            })
            output.append(surface)
            i = link_match.end()
            j += len(surface)
            continue

        # Otherwise, copy character
        output.append(text[i])
        i += 1
        j += 1

    return "".join(output).rstrip('\n'), hyperlinks, section_starts

def process_line(line, redirects, wikipedia_to_qcode, instance_of, wikimedia_internal_classes, disambiguation_qcodes):
    line["text"], line['hyperlinks'], line['section_starts'] = clean_text_with_spans(line["text"])

    clean_hyperlinks = []

    for hyperlink in line['hyperlinks']:
        wiki_page_title = hyperlink['uri'].replace('&amp;', '&').replace('&lt;', '<') \
            .replace('&gt;', '>').replace('&le;', '≤').replace('&ge;', '≥')
        wiki_page_title = wiki_page_title[0].upper() + wiki_page_title[1:]
        if wiki_page_title in redirects:
            wiki_page_title = redirects[wiki_page_title]
        if wiki_page_title in wikipedia_to_qcode:
            qcode = wikipedia_to_qcode[wiki_page_title]
            if qcode in disambiguation_qcodes or (qcode in instance_of and
                                                  len(instance_of[qcode] & wikimedia_internal_classes) > 0):
                # exclude list pages, disambiguation pages, surname pages
                continue
            clean_hyperlinks.append(hyperlink)
            clean_hyperlinks[-1]['qcode'] = qcode
    line['hyperlinks_clean'] = clean_hyperlinks
    return line

def merge_files_and_extract_links(input_dir: str, resources_dir: str, output_dir: str):
    redirects = load_redirects(os.path.join(resources_dir, 'redirects.json'))
    instance_of = load_instance_of(os.path.join(resources_dir, 'instance_of_p31.json'))
    title_to_qcode = load_wikipedia_to_qcode(os.path.join(resources_dir, 'enwiki.json'))

    with open(os.path.join(resources_dir, 'disambiguation_qcodes.txt'), 'r') as f:
        disambiguation_qcodes: Set[str] = {l.rstrip('\n') for l in f.readlines()}

    # list, surnames, redirects, and disambiguation (+ name disambiguation)
    deny_classes = DENY_CLASSES

    processed_clean_wiki_file = open(os.path.join(output_dir, 'wikipedia_links_aligned_sections_spans.json'), 'w')
    pbar = tqdm(total=6e+6)
    for base_path, _, file_names in os.walk(input_dir):
        shuffle(file_names)
        for file_name in file_names:
            with open(os.path.join(base_path, file_name), 'r') as f:
                for line in f:
                    line = json.loads(line)
                    line = process_line(line,
                                        redirects=redirects, wikipedia_to_qcode=title_to_qcode,
                                        instance_of=instance_of, wikimedia_internal_classes=deny_classes,
                                        disambiguation_qcodes=disambiguation_qcodes)
                    processed_clean_wiki_file.write(json.dumps(line) + '\n')
                    pbar.update(1)


if __name__ == '__main__':
    main()

    # text="Female guards in Nazi concentration camps\n\n' (pl. '; ; ) was the position title for a female guard in <a href=\"Nazi%20concentration%20camps\">Nazi concentration camps</a>. Female camp personnel were not official members of the (SS), though they were members of the auxiliary organization that worked under the administration of (SS-TV).\n\nSection::::Background.\nIn April 1933, a <a href=\"Moringen%20concentration%20camp%23%20Female%20camp%2C%20October%201933%20%E2%80%93%20March%201938\">workhouse in Moringen</a> was made into a detention facility under <a href=\"Province%20of%20Hanover\">Hanover</a> administration. In November of that year, 141 women, the majority of whom were suspected or confirmed <a href=\"Communists\">Communists</a>, were imprisoned there. Prisoners usually stayed in Moringen for a few weeks before being released. There have been no reports of mistreatment, and mere group discussions were held daily to \"re-educate\". The facility closed in March 1938, and was replaced by the <a href=\"Lichtenburg%20concentration%20camp\">Lichtenburg concentration camp</a>, which opened in <a href=\"Gau%20Saxony\">Saxony</a> in late 1937, and became known as the first SS-run women's concentration camp. It was commanded by SS- <a href=\"Max%20Koegel\">Max Koegel</a> and staffed by recruited and conscripted women who worked as guards.\n\nOn 1 September 1939, Hitler delivered a speech at the , in which he stated: \"I expect every German woman to integrate herself into the great community-in-struggle in an exemplary fashion and with iron discipline!\" This was the given order despite his the views he held in regards to women: The (BDM) was in support of the 1939 speech and had it written in the organization's 1940 yearbook.\n\nSection::::Recruitment and conscription.\nAdvertisements were posted in newspapers, such as the , which sought out German women between the ages of 20 and 40 to guard women who \"committed an offense against the '<a href=\"Volksgemeinschaft\">Volk community</a>\" at a \"military installation\". For women seeking employment or a higher pay, the job offer was enticing because of the free housing, prepared meals, and the absence of required qualifications. Although a small number of newspaper clippings have survived after the war, history professor Jack G. Morrison claims that the advertisements neglected to mention concentration camps.\n\nIn December 1942, the age range of 20–40 broadened and became 17–45 as tensions grew with the advancement of Allied forces and the 's loss in the <a href=\"Battle%20of%20Stalingrad\">Battle of Stalingrad</a>. During this time, many women were recruited by the labor office, which became a source of contention in postwar testimonies. <a href=\"Johanna%20Langefeld\">Johanna Langefeld</a>, who was an at numerous concentration camps, stated in her testimony: \"There were also cases in which women were sent by one of the labor offices to work as guards at Ravensbrück. This happened most often to women who had refused once or even twice to take the job that had been assigned to them, which meant they were likely to be arrested the next time they refused to take the work assigned to them\".\n\nThe need for female guards in concentration camps became critical when <a href=\"Joseph%20Goebbels\">Joseph Goebbels</a> declared total war against Allied forces in his <a href=\"Sportpalast%20speech\">\"Sportpalast\" speech</a> on 18 February 1943. Hitler raised the age limit for women's involvement to 50 and made employment in <a href=\"Military%20production%20during%20World%20War%20II\">military equipment production</a> mandatory in his 1943 and 1944 decrees pertaining to both male and female participation in the defense of the Reich. The 1943 order exempted individuals who worked at least 48 hours a week, employers of at least five workers, those working in agriculture or health services, pregnant women, and women with one child under the age of six or two children under the age of fourteen. Despite these measures, only a small number of women voluntarily sought out such work, resulting in an increase in SS recruiting and labor office conscriptions.\n\nSection::::Acclimatization and training.\nWhen the <a href=\"Nazi%20Party\">Nazi Party</a> realized that <a href=\"Nazi%20Germany\">Nazi Germany</a> was losing the war, concentration camp personnel destroyed many records, leaving little information regarding how were trained. Ravensbrück has the most preserved records on training practices, largely due to its role as the primary training camp for women from 1942 to 1945.\n\nUpon arrival at Ravensbrück, the recruited and conscripted women were made to sign a slew of documents, including a declaration of confidentiality, a vow not to reprimand prisoners physically or verbally, and an oath of loyalty to Hitler and their superiors. The women were then led to their on-camp quarters. The ensuing training period lasted anywhere from one to six weeks, but as prospects were brought in at an increasing rate, this range decreased to just one week for some women. In June 1942, conscripted Anna David provided testimony of her arrival at Ravensbrück: \n\nA three-month probationary period follows training, during which each prospective is partnered with an experienced who acts as a mentor and is tasked with overseeing a work detail. In 1939, <a href=\"Hermine%20Braunsteiner\">Hermine Braunsteiner</a> received mentoring from <a href=\"Maria%20Mandl\">Maria Mandl</a>, who was then the of Ravensbrück. In a postwar testimony, Braunsteiner states that all were taught how to \"handle, shoot, and clean their service weapon\".\n\nAccording to Commandant's Order No. 3, issued 24 July 1942, new received ideological training every Saturday between five and six o'clock in the evening. As part of the curriculum, two antisemitic Nazi propaganda films were shown, including .\n\nSection::::Membership.\nDue to missing and destroyed documentation pre-liberation, the exact number of women who became between 1938 and 1945 has been disputed. Based on published literature and surviving evidence from numerous concentration camps, it is estimated that 3,500 women served as guards. Historian has further broken down this figure, estimating that 313 women were employed at Ravensbrück as camp personnel in late 1942 from payroll records. By late 1944, the total surpassed 3,000. \n\nSection::::Ranks and uniforms.\nBraunsteiner has claimed that the first group of women assigned to Ravensbrück were only given blue smocks to wear. About a year later, prototype uniforms were supplied. In the first design, a light gray <a href=\"loden%20cloth\">loden cloth</a> jacket and <a href=\"culottes\">culottes</a> were worn with a blue blouse, black boots, and a light gray <a href=\"side%20cap\">side cap</a>. Only after Himmler's visit to Ravensbrück in the spring of 1940 did receive standardized uniforms. Two military gray uniforms, one for winter and one for summer, were provided, together with two pairs of boots, blouses, a cap, and sportswear. Hats and jewelry were prohibited, with the exception of the designated side cap or a straw hat on hot days.\n\nTo denote rank, uniforms displayed aluminum braiding on the shoulders and sleeves, as well as badges and awards such as the <a href=\"War%20Merit%20Medal\">War Merit Medal</a> Second Class.\n\nSection::::Aufseherin.\n means \"female SS overseer\". were in charge of conducting the daily roll call, or , allocating inmates to work details, and guarding prisoners.\nSection::::\n means \"female dog handler\". There is little documentation on female dog handlers in concentration camps, with the only known name being that of <a href=\"Elfriede%20Rinkel\">Elfriede Rinkel</a>, though it is assumed that they had similar training and responsibilities to their male counterparts. Himmler allegedly ordered for to not carry guns, though this is refuted by Braunsteiner's testimony, so some women were armed with German shepherds who Himmler demanded to be \"trained to savage to death anyone except their handler\".\nSection::::\n means \"commanding officer\", though the women with this title were just in charge of overseeing certain work details in a concentration camp.\nSection::::\n and means \"block leader\". The terms , meaning \"block senior\" and blockova were the titles given to prisoners if they, like the appointed , were put in charge of maintaining order within their respective block.\nSection::::\n means \"labor service leader\". These women were in charge of assigning work details amongst the prisoners, maintaining efficiency within the concentration camp, and overseeing .\nSection::::\n means \"report leader\". coordinated daily schedules and work schedules from an office within the camp and received reports from other guards regarding any incidents, illnesses, and deaths.\nSection::::\n means \"first supervisor\".\n\nSection::::\n means \"female chief senior overseer\".\nSection::::\n, commonly shortened in literature as , means \"camp leader\". dealt with affairs concerning the prisoners at the concentration camp, coordinated with the Labor Squad office to appoint work details to prisoners, and worked closely with the subordinate .\n\nSection::::Daily life.\nSection::::Housing.\n were housed at Ravensbrück based on their rank. Since personal and intimate contact with the opposite sex was prohibited, the eight apartments on the campgrounds were all far from the men's quarters. Only young and unmarried guards were placed in these apartments, each of which were two stories with ten bedrooms and four attic rooms. Each building is believed to have held at least 112 women. Private housing arrangements were made for married women and mothers. Despite the policy that male and female camp personnel be separated on camp grounds, it remained a problem, with Maria Mandl and <a href=\"Dorothea%20Binz\">Dorothea Binz</a> engaging in their own liaisons while employed there.\n\nSection::::Recreation.\n were allowed to leave Ravensbrück only on specific days and with a curfew of 11 P.M., which a number of them ignored. During the spring and summer, the women frequented movie theaters, pubs, and festivals. If they remained in campgrounds, free time was spent sewing or getting their hair done at the prisoner-run salon.\n\nThe women did not have to do their own laundry, cleaning, or cooking as prisoners were made to do it for them. Some considered this a luxury. <a href=\"Herta%20Ehlert\">Herta Ehlert</a> stated in her postwar testimony: \"Well, I want to be quite honest, I had never such a good life as in the beginning at Ravensbrück when I arrived\".\n\nSection::::Trials and sentences.\nSection::::Majdanek trials.\nElsa Ehrich was the first and only woman to face a death sentence in the second Majdanek trial, which took place between 1946 and 1948. Between 1975 and 1981, Alice Orlowski, Hermine Braunsteiner (life imprisonment), Hildegard Lächert (12 years imprisonment), and Hermine Böttcher Brückner (acquitted and released) appeared in court for the third trial. \n\nOrlowski died of natural causes during court proceedings, but had first been tried in the 1947 Kraków Auschwitz trial and received a sentence of fifteen years in prison. Witnesses identified Braunsteiner based on the War Merit Medal, which she wore every day on her jacket whilst working at the Majdanek camp. Conversely, Böttcher Brückner was pinned by survivors at the Majdanek trial as being \"good\" and \"humane\", in comparison to other , though she had struck the prisoners from time to time.\n\nSection::::Belsen trials.\nThe first Belsen trial took place in 1945, in which Irma Grese, Elisabeth Volkenrath, and Johanna Bormann received a sentence of death by hanging. Herta Bothe, Hilde Lobauer, and Irene Haschke received prison sentences of ten years, whilst Herta Ehlert received fifteen years. Gertrud Heise and Anneliese Kohlmann were only sentenced the following year during the second Belsen trial, receiving fifteen years and two years' imprisonment, respectively.\n\nSection::::Stutthof trials.\nJenny-Wanda Barkmann, Elisabeth Becker, Wanda Klaff, Ewa Paradies, and Gerda Steinhoff all received a sentence of death by hanging in the first Stuffhof trial, which took place Gdańsk, Poland in 1946. Erna Beilhardt had been the only woman to not receive a death sentence, having only received five years in prison.\n\nSection::::Kraków Auschwitz trial.\nThe 1947 Auschwitz trial in Kraków, Poland sentenced Maria Mandl and Therese Brandl to death by hanging. Luise Danz was sentenced to life in prison. Alice Orlowski and Hildegard Lächert were sentenced to fifteen years in prison.\n\nSection::::Aftermath.\nSection::::Perpetrators postwar.\nOne of the few former to tell her story to the public was Hertha Bothe, who had been employed at Ravensbrück in 1942, then at Stutthof and its <a href=\"Bromberg-Ost\">Bromberg-Ost</a> subcamp, Auschwitz, and Bergen-Belsen. She was given early release in the mid-1950s from her ten-year prison sentence. In an interview recorded in 1999, Bothe was asked if she regretted being a concentration camp guard. She replied, \"Did I make a mistake? No. The mistake was that it was a concentration camp, but I had to go. Otherwise, I would have been put into it myself; that was my mistake\". Though Bothe claimed that refusal of the job would have resulted in her own arrest—an explanation given by many former —it was unlikely to have been true, as surviving records have shown that the new recruits refusing to remain as guards in Ravensbrück did not face consequences.\n\nFormer Ravensbrück Elfriede Rinkel was eighty-four and living in San Francisco when she was deported to Germany by the U.S. Justice Department in August 2006. She kept her participation in the Nazi Party a secret from her family, friends, and Jewish-German husband of forty-two years, Fred. She had emigrated to the United States in 1959 in search of a better life, and had omitted Ravensbrück from the list of residences on her visa application. Rinkel ultimately faced no criminal charges in Germany, as the statute of limitations had expired. The case continued to be examined until her death in 2018.\n\n\n\n\n\n"
    # clean_text_with_spans(text)
