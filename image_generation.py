from captcha.image import ImageCaptcha
import string
import random
import uuid
import glob
import datetime
import os

class ImageGenerator():
    def __init__(self, vocab=string.digits):
    
        self._base_dir_name = "generated_images_{dt}".format(dt=datetime.datetime.now().strftime("%s"))
        # Create the base dir we're gonna save our generated images in
        os.mkdir(self._base_dir_name)
        
        # self.captcha_generator = ImageCaptcha(fonts=self._fonts, width=80, height=80)
        self.captcha_generator = ImageCaptcha(width=128, height=128)
#         self.vocab = string.ascii_letters + string.digits
        # self.vocab = string.ascii_lowercase
#         self.vocab = 'ow' #only train for the o & w letters
        # self.vocab = string.digits
        # self.vocab = vocab
        self.vocab = "01"

        self.n_min_chars = 4
        self.n_max_chars = 6
        
    def generate_captcha_sequence(self):
        """
        Randomly create a CAPTCHA sequence
        """
        captcha = ""
        captcha_length = random.randint(self.n_min_chars, self.n_max_chars)
        for x in range(captcha_length):
            captcha += random.choice(self.vocab)

        return captcha
    
    def generate_captcha_image(self, sequence):
        """
        Receive a CAPTCHA sequence (string), creates a CAPTCHA image, and returns the generated file name
        """
        random_id = str(uuid.uuid4())[-8:]
        filename = "./{dirname}/{sequence}_{random_id}.png".format(dirname=self._base_dir_name, sequence=sequence, random_id=random_id)
        self.captcha_generator.write(sequence, filename)
        return filename
    
    def execute_img_generation(self, num_of_images=10):
        """
        Generate N CAPTCHA images, N received as a parameter.
        Each image is saved as a PNG file
        """
        for i in range(num_of_images):
            print("Running generation number #{i}".format(i=i+1))
            sequence = self.generate_captcha_sequence()
#             print("Selected char {c}".format(c=char))
            fname = self.generate_captcha_image(sequence)
#             print("Wrote to {p}".format(p=fname))

ig = ImageGenerator()
ig.execute_img_generation()