import requests
from dotenv import load_dotenv
import os
import base64
load_dotenv()



def ocr_space_file(filename, overlay=True, api_key=f'{os.getenv("key")}', language='eng'):
    """ OCR.space API request with local file.
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               'filetype': 'pdf',
               'isCreateSearchablePdf': True,
               'OCREngine': 1
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f },
                          data=payload,
                          )
        
    return r.content.decode()


def ocr_space_url(url, overlay=False, api_key=f'{os.getenv("key")}', language='eng'):
    """ OCR.space API request with remote file.
    :param url: Image url.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'url': url,
               'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               'filetype': 'SVG',
               'isCreateSearchablePdf': False
               }
    r = requests.post('https://api.ocr.space/parse/image',
                      data=payload,
                      )
    return r.content.decode()


# Use examples:
test_file = ocr_space_file(filename='file.pdf')
print(test_file)
# test_url = ocr_space_url(url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTHO58Dsh3jN_L_eVPqpyI94VRq8OIoOie1Ig&s')
# print(test_url)
