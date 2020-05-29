from captcha.image import ImageCaptcha
import numpy as np
import string
from PIL import Image
import random

class CaptchaGenerator(object):
    max_char_set = ['digits','letters']
    def __init__(self,char_set=['digits'],lengths=[4,5,6,7],shape=(60,160)):
        assert set(char_set).issubset(CaptchaGenerator.max_char_set)\
            and char_set != [],'char_set 参数错误!'
        self.char_set = char_set
        self.generate_chars()
        self.lengths = lengths
        self.maxlen = max(lengths)
        self.shape = shape
        self.captcha = ImageCaptcha(height=self.shape[0],
                                    width=self.shape[1])
    def generate_chars(self):
        s = ''
        if 'digits' in self.char_set:
            s += string.digits
        if 'letters' in self.char_set:
            s += string.ascii_lowercase
        self.chars = list(s)
    def random_text(self):
        text = ''
        label = []
        length = random.choice(self.lengths)
        for i in range(length):
            n = random.randint(0,len(self.chars)-1)
            if n>=10 and random.random()>0.5:
                text += self.chars[n].upper()
            else:
                text += self.chars[n]
            label.append(n)
        return text,label
    def generate_image(self,text):
        image = self.captcha.generate(text)
        image = Image.open(image)
        image = np.array(image)
        return image
    def create_sample(self):
        text,label = self.random_text()
        image = self.generate_image(text)
        return image,label,text
    def __call__(self):
        while True:
            image,label,_ = self.create_sample()
            seqlen = len(label)
            label.extend([-1]*(self.maxlen-seqlen))
            label = np.array(label)
            yield image,label,seqlen
        return