from PIL import Image

# from scorer_wrapper_lib import evaluate as scorer_evaluate
from tagger_wrapper_lib import evaluate as tagger_evaluate

scorer = ["ImprovedAesthetic"]
tagger = "GITLargeCaptioning"

image = Image.open("tests/resources/img/1_img/file05.webp")
# results = scorer_evaluate([image], scorer)
# print(f"scorer_evaluate: {results}")

results2 = tagger_evaluate([image], [tagger])
print(f"tagger_evaluate: {results2}")
# print(f"tagger_evaluate: {results2[tagger][0]['annotation']}")
