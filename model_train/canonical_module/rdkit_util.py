import re

# Bring charge appears in the middle of brackets to tail
# so that SMILES parser in RDkit can parse correctly (e.g. [O-H] -> [OH-])
def bringChargeToTail(smiles):
    pattern = '(.*)\[([^\[\]]+)([\+\-]\d*)(\D[^\[\]]*)\](.*)'
    while True:
        matched = re.search(pattern, smiles)
        if matched:
            smiles = "{}[{}{}{}]{}".format(matched.group(1), matched.group(2), matched.group(4), matched.group(3), matched.group(5))
            print(smiles)
        else:
            break
    return smiles
