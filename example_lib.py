from PIL import Image
from scorer_wrapper_lib import evaluate

model_name = ["ImprovedAesthetic", "WaifuAesthetic"]

image = Image.open("tests/resources/img/1_img/file05.webp")
results = evaluate([image], model_name)
print(f"results: {results}")

results2 = evaluate([image], model_name)
print(f"results2: {results2}")
