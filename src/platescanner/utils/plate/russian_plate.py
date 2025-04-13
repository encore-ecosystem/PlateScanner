import re


def process_text(recognized_text: str) -> str:
    recognized_text = re.sub(r"[^A-Za-z0-9]", "", recognized_text).upper()
    recognized_text = recognized_text.replace('V', 'Y')
    recognized_text = recognized_text.replace('I', '')
    recognized_text = recognized_text.replace('R', 'B')
    recognized_text = recognized_text.replace('G', 'C')
    recognized_text = recognized_text.replace('F', 'A')
    recognized_text = recognized_text.replace('D', 'O')
    recognized_text = recognized_text.replace('S', '5')
    recognized_text = recognized_text.replace('Z', '2')
    recognized_text = list(recognized_text)
    if len(recognized_text) == 0:
        return ""
    match recognized_text[0]:
        case "8":
            recognized_text[0] = "B"
        case "0":
            recognized_text[0] = "O"
        case "7":
            recognized_text[0] = "T"
    for i in range(1, 4):
        if i >= len(recognized_text):
            break
        match recognized_text[i]:
            case "B":
                recognized_text[i] = "8"
            case "O":
                recognized_text[i] = "0"
            case "T":
                recognized_text[i] = "7"
    for i in range(4, 6):
        if i >= len(recognized_text):
            break
        match recognized_text[i]:
            case "8":
                recognized_text[i] = "B"
            case "0":
                recognized_text[i] = "O"
            case "7":
                recognized_text[i] = "T"
    recognized_text = "".join(recognized_text)
    if len(recognized_text) >= 9:
        recognized_text = recognized_text[:9]
    return recognized_text.strip()


__all__ = [
    'process_text'
]
