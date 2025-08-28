import base64, requests

with open("/home/ubuntu/wenjinf-workspace/EgoBlur/test_data/img_333_1751995527_515604Z.jpg","rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

r = requests.post("http://127.0.0.1:1234/blur", json={"image": img_b64, "quality": 90})
r.raise_for_status()
out_b64 = r.json()["blurred_image"]

with open("/home/ubuntu/wenjinf-workspace/EgoBlur/output/blurred.jpg","wb") as f:
    f.write(base64.b64decode(out_b64))
print("Saved blurred.jpg")