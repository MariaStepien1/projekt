from urllib import request

import cv2
from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


class PeopleCounter(Resource):
    def get(self):
        img = cv2.imread('zdjproj.jpg')
        boxes, weights = hog.detectMultiScale(img, winStride=(4, 4))

        return {'count': len(boxes)}

class PeopleCounterWithLinkGiven(Resource)
    def get(self):
        url = request.args.get('url')
        if url:
            img2 = cv2.imread(url)
            boxes, weights = hog.detectMultiScale(img2, winStride=(5, 5))
        else:
            return{'error': 'url parameter not provided'}

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(PeopleCounter, '/')
api.add_resource(HelloWorld, '/test')

if __name__ == '__main__':
    app.run(debug=True)