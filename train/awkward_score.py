from collections import Counter

def get_response_ratio(dialogue):
    speakers = [msg.split(":")[0].strip() for msg in dialogue if ":" in msg]
    counter = Counter(speakers)
    if not counter or sum(counter.values()) == 0:
        return 0
    most_common_ratio = max(counter.values()) / sum(counter.values())
    return round(most_common_ratio, 2)

acknowledgement_words = {"ㅇㅇ", "넹", "아", "그렇구나", "응", "오"}
reaction_words = {"헐", "ㅋㅋ", "ㅎㅎ", "와", "우와", "엥"}

def get_ack_react_ratio(dialogue):
    ack_count = sum(1 for msg in dialogue if msg.strip() in acknowledgement_words)
    react_count = sum(1 for msg in dialogue if msg.strip() in reaction_words)
    return (ack_count + react_count) / len(dialogue)