"""Task-area mappings and dataset metadata for supported benchmarks."""

from __future__ import annotations

from typing import NamedTuple

TWEET_EMOTION_CLASSIFICATION = "tweet_emotion_classification"
SCHOOL_MATH_REASONING = "school_math_reasoning"
CONCEPT_TO_SENTENCE_GENERATION = "concept_to_sentence_generation"
CONTEXT_QUESTION_ANSWERING = "context_question_answering"
TEXT_SUMMARIZATION = "text_summarization"

SUPPORTED_TASK_AREAS = (
    TWEET_EMOTION_CLASSIFICATION,
    SCHOOL_MATH_REASONING,
    CONCEPT_TO_SENTENCE_GENERATION,
    CONTEXT_QUESTION_ANSWERING,
    TEXT_SUMMARIZATION,
)

TASK_AREA_TO_DATASET: dict[str, str] = {
    TWEET_EMOTION_CLASSIFICATION: "tweeteval",
    SCHOOL_MATH_REASONING: "gsm8k",
    CONCEPT_TO_SENTENCE_GENERATION: "common_gen",
    CONTEXT_QUESTION_ANSWERING: "squad_v2",
    TEXT_SUMMARIZATION: "xsum"
}

DATASET_LABEL_SETS: dict[str, set[str]] = {"tweeteval": {"anger", "joy", "optimism", "sadness"}}


class Example(NamedTuple):
    """A single real (input, target) pair used to ground TaskSpec generation for a dataset."""

    input: str
    target: str


DATASET_EXAMPLES: dict[str, tuple[Example, ...]] = {
    "common_gen": (
        Example(
            input="lake, shore, canoe",
            target="A canoe on shore with rainbow across the lake",
        ),
        Example(
            input="boat, lake, drive",
            target="The fisherman drives his boat on the lake",
        ),
        Example(
            input="grass, horse, eat",
            target="In the field, a horse eats the grass.",
        ),
    ),

    "gsm8k": (
        Example(
            input="On a school trip to the seashore, Alan and his friends collected shells. "
                  "Alan collected four times as many shells as Ben did. "
                  "Ben got a late start and only collected a third of what Laurie did. "
                  "If Laurie collected 36 shells how many did Alan collect?",
            target="48",
        ),

        Example(
            input=(
                "A robe takes some bolts of blue fiber and half that much white fiber. "
                "There are 3 bolts in total. How many blue fibers are there?"
            ),
            target=(
                "2"
            ),
        ),

        Example(
            input=(
                "Sam memorized six more digits of pi than Carlos memorized. "
                "Mina memorized six times as many digits of pi as Carlos memorized. "
                "If Mina memorized 24 digits of pi, how many digits did Sam memorize?"
            ),
            target=(
                "10"
            ),
        ),
    ),

    "tweeteval": (
        Example(
            input="“Worry is a down payment on a problem you may never have'. "
                  "Joyce Meyer.  #motivation #leadership #worry",
            target="optimism",
        ),
        Example(
            input="it's pretty depressing when u hit pan on ur favourite highlighter",
            target="sadness",
        ),
        Example(
            input="No but that's so cute. Atsu was probably shy about photos before but cherry helped her out uwu",
            target="joy",
        ),
        Example(
            input="Rooneys fucking untouchable isn't he? Been fucking dreadful again, depay has looked decent(ish)tonight",
            target='anger',
        ),

    ),
    "squad_v2": (
        Example(
            input='Context: The Roman Catholic Church canon law also includes the main five rites (groups) of '
                  'churches which are in full union with the Roman Catholic Church and the Supreme Pontiff:'
                  'Question: What term characterizes the intersection of the rites with the Roman Catholic Church?',
            target='full union',
        ),
        Example(
            input='Context: Machine languages and the assembly languages that represent them '
                  '(collectively termed low-level programming languages) tend to be unique to a particular type '
                  'of computer. For instance, an ARM architecture computer '
                  '(such as may be found in a PDA or a hand-held videogame) cannot understand the machine language of '
                  'an Intel Pentium or the AMD Athlon 64 computer that might be in a PC.'
                  'Question: An ARM architecture computer can be found in what?',
            target='a PDA or a hand-held videogame',
        ),
        Example(
            input='Context: Many of the instruments used to perform medieval music still exist, but in different forms. '
                  'Medieval instruments included the wood flute (which in the 21st century is made of metal), '
                  'the recorder and plucked string instruments like the lute. As well, early versions of the organ, '
                  'fiddle (or vielle), and trombone (called the sackbut) existed. '
                  'Medieval instruments in Europe had most commonly been used singly, often self accompanied with '
                  'a drone note, or occasionally in parts. From at least as early as the 13th century through '
                  'the 15th century there was a division of instruments into haut (loud, shrill, outdoor instruments) '
                  'and bas (quieter, more intimate instruments).'
                  'Question: What was the medieval flute made from?',
            target='wood',
        ),
    ),
    "xsum": (
        Example(
            input='The theme tune of Antiques Roadshow was played as the presenter\'s coffin was carried out '
                  'of the church at Mawnan Smith near Falmouth.\nScully joined the BBC as a freelance journalist '
                  'in 1965 and hosted the BBC\'s Nationwide before presenting Antiques Roadshow with Arthur Negus '
                  'from 1981.\nThe presenter\'s family described the funeral as "a wonderful occasion".'
                  '\nA lot of people thought he was the Antiques Roadshow and will never get used to anyone else '
                  'presenting it\nScully hosted the BBC\'s Nationwide before presenting Antiques Roadshow with '
                  'Arthur Negus from 1981.\nHe resigned from the BBC One show in 2000 to join an internet auction '
                  'company launching an antiques business.\nThe presenter\'s eldest son Charles Scully told the '
                  'BBC his father\'s success was partly due to his "ability to put people at ease".\n'
                  'He said: "His ability to talk to everybody from a shopkeeper to a president will be sadly missed."'
                  '\nFormer Nationwide presenter Sue Lawley remembered Scully as a "great talent" who was "fun-loving" '
                  'and most proud of his interviews with Margaret Thatcher.',
            target='The funeral has been held for the former Antiques Roadshow TV host Hugh Scully, '
                   'who died at the age of 72.',
        ),
        Example(
            input='Up to 100,000 youngsters will be eligible for half-price day tickets using The Young Persons '
                  '16-18 card from September.\nIt was agreed by the area\'s mayor Andy Burnham and Transport for '
                  'Greater Manchester, and a similar scheme is being considered for the Metrolink.\nHajrah Ahmed, 17, '
                  'said half-price bus tickets "will be such a big help".\nThe Manchester College business student'
                  ' who travels to Openshaw from Cheetham Hill every day said her journeys are costing £100 per month.'
                  '\n"[It] is obviously an awful lot of money for someone like me, who doesn\'t have a part-time job.'
                  '\n"I can look ahead to the next year or so without the worry of how much money I am spending on my '
                  'journey," she said.\nThe deal was proposed by Mr Burnham in his manifesto for mayor in April.\n"I '
                  'promised to help our young people get on in life, and this is the first step in delivering on '
                  'that," Mr Burnham said.\nGreater Manchester Travelcards Ltd, which represents all bus companies '
                  'in the area, will extend its multi-operator 50% discounted 16-and-under ticket.\nA junior day ticket'
                  ' to cover 16 to 18 year olds will also be introduced.\nEligibility to use the ticket will run up '
                  'to 31 August after the user\'s 18th birthday.',
            target='Discounted bus tickets for 16 to 18 year olds will be rolled out in Greater Manchester, '
                   'it has been announced.',
        ),
        Example(
            input='Ogilvie, 21, has yet to make a first team appearance for Spurs and spent most of the last two '
                  'seasons on loan at League Two Stevenage.\nThe former under-16 and under-17 England international '
                  'made 18 appearances for the Boro last season.\n"I\'m looking forward to it and I want to be playing '
                  'games regularly," Ogilvie told the club website.\n"I\'m really pleased to secure Connor\'s signature. '
                  'He\'s got pedigree having come through the youth ranks at Tottenham and what is an added bonus for '
                  'us is that he has experience of playing league football," added Gillingham manager Ady Pennock.'
                  '\nFind all the latest football transfers on our dedicated page.',
            target='League One side Gillingham have signed Tottenham Hotspur defender '
                   'Connor Ogilvie on a six-month loan deal.',
        ),
    ),
}
