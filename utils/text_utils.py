import re


def clean_text(text: str) -> str:
    """
    Cleans transcript text:
    - removes filler words
    - removes extra spaces
    - lowercases text
    """

    # remove filler words (basic)
    fillers = ["um", "uh", "like", "you know"]
    for word in fillers:
        text = re.sub(rf"\b{word}\b", "", text, flags=re.IGNORECASE)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()