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
    TEXT_SUMMARIZATION: "xsum",
}

DATASET_LABEL_SETS: dict[str, set[str]] = {
    "tweeteval": {"anger", "joy", "optimism", "sadness"}
}


class Example(NamedTuple):
    """A single real (input, target) pair used to ground TaskSpec generation for a dataset."""

    input: str
    target: str


DATASET_EXAMPLES: dict[str, tuple[Example, ...]] = {
    "common_gen": (
        Example(
            input="['dog', 'leap', 'catch']",
            target="A dog leaps into the air to catch a frisbee.",
        ),
        Example(
            input="['chef', 'slice', 'tomato', 'knife']",
            target="Using a sharp knife, the chef slices a tomato for the salad.",
        ),
        Example(
            input="['cat', 'hide', 'box']",
            target="A cat hides inside an empty cardboard box.",
        ),
        Example(
            input="['child', 'feed', 'duck', 'pond']",
            target="Beside the pond, a child crouches down to feed the ducks.",
        ),
        Example(
            input="['cyclist', 'push', 'bicycle', 'hill', 'rain']",
            target="Caught in the rain, a cyclist pushes her bicycle up a muddy hill.",
        ),
    ),
    "gsm8k": (
        Example(
            input=(
                "On a school trip to the seashore, Alan and his friends collected shells. "
                "Alan collected four times as many shells as Ben did. "
                "Ben collected a third as many shells as Laurie did. "
                "If Laurie collected 36 shells, how many shells did Alan collect?"
            ),
            target="48",
        ),
        Example(
            input=(
                "A robe requires some bolts of blue fiber and half as many bolts "
                "of white fiber. There are 3 bolts in total. "
                "How many bolts of blue fiber are needed?"
            ),
            target="2",
        ),
        Example(
            input=(
                "Sam memorized six more digits of pi than Carlos memorized. "
                "Mina memorized six times as many digits of pi as Carlos memorized. "
                "If Mina memorized 24 digits, how many digits did Sam memorize?"
            ),
            target="10",
        ),
        Example(
            input=(
                "Maya buys 4 notebooks for $3 each and 2 pens for $2 each. "
                "She pays with a $20 bill. How many dollars in change does she receive?"
            ),
            target="4",
        ),
        Example(
            input=(
                "A bus travels 45 miles per hour for 2 hours and then "
                "30 miles per hour for 1 hour. How many miles does it travel in total?"
            ),
            target="120",
        ),
        Example(
            input=(
                "A jacket originally costs $80. The store gives a 25 percent discount. "
                "How many dollars does the jacket cost after the discount?"
            ),
            target="60",
        ),
        Example(
            input=(
                "A library has 250 books. It lends out 68 books on Monday "
                "and 47 books on Tuesday. Then 25 books are returned. "
                "How many books are in the library now?"
            ),
            target="160",
        ),
        Example(
            input=(
                "A bakery makes 72 cupcakes. It packs 6 cupcakes in each box. "
                "After selling 5 boxes, how many cupcakes remain?"
            ),
            target="42",
        ),
    ),
    "tweeteval": (
        Example(
            input=(
                "@user yeah thanks for cancelling it AFTER we all got there 🙃 "
                "TWO HOURS wasted for absolutely nothing #brilliant"
            ),
            target="anger",
        ),
        Example(
            input=(
                "How do you lose my order TWICE and then tell me to 'just place "
                "another one'?? 😂 WHAT A JOKE"
            ),
            target="anger",
        ),
        Example(
            input=(
                "@user love how you can ignore every message for a WEEK then suddenly "
                "need an answer from me RIGHT NOW lol #nice"
            ),
            target="anger",
        ),
        Example(
            input=(
                "@user nah it's FINE, you guys have fun :) kinda getting used to "
                "finding out about everything from the photos anyway"
            ),
            target="sadness",
        ),
        Example(
            input=(
                "Still catch myself saving things to send you and then remembering "
                "there's NOBODY on the other end anymore."
            ),
            target="sadness",
        ),
        Example(
            input=(
                "@user you absolute idiot 😂❤️ can't believe you travelled ALL THAT WAY "
                "just to surprise me, I'm still smiling"
            ),
            target="joy",
        ),
        Example(
            input=(
                "@user ONE rejection doesn't decide where this goes. send the next "
                "application, then the next one. somebody's gonna say YES #keepgoing"
            ),
            target="optimism",
        ),
    ),
    "squad_v2": (
        Example(
            input="The economy of Victoria is highly diversified: service sectors including financial and property "
            "services, health, education, wholesale, retail, hospitality and manufacturing constitute the "
            "majority of employment. Victoria's total gross state product (GSP) is ranked second in Australia, "
            "although Victoria is ranked fourth in terms of GSP per capita because of its limited mining "
            "activity. Culturally, Melbourne is home to a number of museums, art galleries and theatres and is "
            'also described as the "sporting capital of Australia". The Melbourne Cricket Ground is '
            "the largest stadium in Australia, and the host of the 1956 Summer Olympics and the 2006 "
            'Commonwealth Games. The ground is also considered the "spiritual home" of Australian cricket '
            "and Australian rules football, and hosts the grand final of the Australian Football League (AFL) "
            "each year, usually drawing crowds of over 95,000 people. Victoria includes eight public "
            "universities, with the oldest, the University of Melbourne, having been founded in 1853. What "
            "city in Victoria is called the sporting capital of Australia?",
            target="Melbourne",
        ),
        Example(
            input="In the course of the 10th century, the initially destructive incursions of Norse war bands into "
            "the rivers of France evolved into more permanent encampments that included local women and "
            "personal property. The Duchy of Normandy, which began in 911 as a fiefdom, was established by "
            "the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking "
            "ruler Rollo, and was situated in the former Frankish kingdom of Neustria. The treaty offered Rollo "
            "and his men the French lands between the river Epte and the Atlantic coast in exchange for their "
            "protection against further Viking incursions. The area corresponded to the northern part of "
            "present-day Upper Normandy down to the river Seine, but the Duchy would eventually extend west "
            "beyond the Seine. The territory was roughly equivalent to the old province of Rouen, and "
            "reproduced the Roman administrative structure of Gallia Lugdunensis II "
            "(part of the former Gallia Lugdunensis). When was the Duchy of Normandy founded?",
            target="911",
        ),
    ),
    "xsum": (
        Example(
            input=(
                "A fire broke out overnight at a warehouse on the outskirts of Bristol, "
                "forcing nearby residents to leave their homes. More than 60 firefighters "
                "attended the scene and roads around the industrial estate were closed. "
                "The fire service said no injuries had been reported and investigators "
                "were working to determine the cause."
            ),
            target=(
                "Residents were evacuated after a large warehouse fire broke out "
                "on the outskirts of Bristol."
            ),
        ),
        Example(
            input=(
                "The city council approved plans for a new sports centre after months of "
                "debate over its cost. The £28m complex will include a swimming pool, gym "
                "and indoor courts. Opposition councillors criticised the budget, while "
                "local sports clubs welcomed the decision. Construction is expected to "
                "begin next spring."
            ),
            target=(
                "The city council has approved a £28m sports centre that is due "
                "to begin construction next spring."
            ),
        ),
        Example(
            input=(
                "Maya Lewis joined the museum as an assistant curator in 2004 and later "
                "led several major exhibitions. She became director in 2016 and oversaw "
                "a major expansion of the modern-art collection. The museum announced on "
                "Tuesday that Lewis will step down at the end of the year to become head "
                "of the National Arts Foundation."
            ),
            target=(
                "Museum director Maya Lewis will step down at the end of the year "
                "to lead the National Arts Foundation."
            ),
        ),
        Example(
            input=(
                '"This is a disappointing day for everyone involved," said manager '
                "Daniel Price after Westford lost 2-1 to Harborough. Westford had taken "
                "the lead in the first half but conceded twice after the break. The defeat "
                "means they will miss the play-offs for the first time in five seasons."
            ),
            target=(
                "Westford will miss the play-offs for the first time in five seasons "
                "after losing 2-1 to Harborough."
            ),
        ),
        Example(
            input=(
                "Researchers at Northbridge University tested a new battery material over "
                "18 months. Early trials showed improved charging speed, although the team "
                "said more work was needed on long-term durability. The researchers have "
                "now demonstrated that the material can retain 90% of its capacity after "
                "1,000 charging cycles."
            ),
            target=(
                "Northbridge University researchers have developed a battery material "
                "that retained 90% of its capacity after 1,000 charging cycles."
            ),
        ),
        Example(
            input=(
                "The government announced a review of rural transport funding following "
                "complaints from local councils. Several councils said recent cuts had "
                "left villages with fewer bus services. Ministers said the review would "
                "report later this year. Separately, the government confirmed that £40m "
                "would be made available immediately to protect existing rural routes."
            ),
            target=(
                "The government has announced £40m in immediate funding to protect "
                "rural bus routes."
            ),
        ),
        Example(
            input=(
                "Singer Lena Brooks began her career performing in small clubs before "
                "releasing her first album in 1998. She later won three national music "
                "awards and toured internationally. Her latest album was released last "
                "year. Brooks has announced that she will retire from touring after a "
                "final series of concerts next summer."
            ),
            target=(
                "Singer Lena Brooks will retire from touring after a final series "
                "of concerts next summer."
            ),
        ),
        Example(
            input=(
                "Rovers dominated possession for much of the match and created several "
                "chances before half-time. Their captain missed a penalty in the 63rd "
                "minute, but substitute Aaron Cole scored with five minutes remaining. "
                "The 1-0 victory secured Rovers promotion to the top division for the "
                "first time in 12 years."
            ),
            target=(
                "Rovers have won promotion to the top division for the first time "
                "in 12 years after beating their opponents 1-0."
            ),
        ),
    ),
}
