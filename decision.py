def make_decision(objects):

    if len(objects) == 0:
        return "No Object"

    for obj in objects:
        label, conf, distance = obj

        if label == "person" and distance < 100:
            return "STOP"

        elif label == "car" and distance < 150:
            return "Slow Down"

    return "Go Straight"