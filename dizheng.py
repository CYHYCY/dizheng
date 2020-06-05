from flask import Flask, request, Response
import yaml
import json
from utils.base64_util import base64_PIL_BGR
from detect_image import face_detect

##### flask #####
responseTemplete = {"code": "10000", "msg": "成功。"}
app = Flask(__name__)
errorsPath = "errors.yaml"
with open(errorsPath, 'r', encoding="utf-8") as errorsFile:
    errors = yaml.safe_load(errorsFile)

##### dizheng #####
model = face_detect()


@app.route("/rimmindplus/dizheng/face_detect", methods=['POST'])
def isPorn():
    response = responseTemplete.copy()
    body = request.get_json()
    base64Image = body.get("image")
    if base64Image == None:
        response['code'] = errors.get("10715")
        response['msg'] = errors.get("10715-description")
        return Response(json.dumps(response))
    try:
        image = base64_PIL_BGR(base64Image)
    except BaseException as e:
        response['code'] = errors.get("10001")
        response['msg'] = errors.get("10001-description")
        return Response(json.dumps(response))
    results = model.detect(image)
    response['data'] = {'result': results}
    return Response(json.dumps(response))


if __name__ == "__main__":
    app.run(threaded=True, debug=False, host="0.0.0.0", port=8000)
