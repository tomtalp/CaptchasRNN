from PIL import Image

from random import choices
from captcha.image import ImageCaptcha
from claptcha import Claptcha
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
        
        self.font = "/Users/tomtalpir/dev/tom/captcha_project/CaptchaImgGeneration/fonts/Seravek.ttc"
        self.img_w = 160
        self.img_h = 160

        self.captcha_generator = ImageCaptcha(width=self.img_w, height=self.img_h, fonts=[self.font])
        self.vocab = string.ascii_lowercase + string.digits
        # self.vocab = string.ascii_lowercase

        self.n_min_chars = 4
        self.n_max_chars = 8
        
    def generate_captcha_sequence(self):
        """
        Randomly create a CAPTCHA sequence
        """
        captcha = ""
        captcha_length = random.randint(self.n_min_chars, self.n_max_chars)
        for x in range(captcha_length):
            captcha += random.choice(self.vocab)

        return captcha
    
    def select_captcha_generator(self):
        """
        Select the type of Captcha generator we'll use
        1 = PyCaptcha, Pr(x) = 0.65
        2 = Claptcha, Pr(x) = 0.35
        """
        population = [1, 2]
        # weights = [0.65, 0.35]
        weights = [0, 0.1]

        return choices(population, weights)[0]
    
    def generate_pycaptcha_image(self, sequence):
        """
        Receive a CAPTCHA sequence (string), creates a CAPTCHA image, and returns the generated file name.
        Using PyCaptcha to generate
        """
        random_id = str(uuid.uuid4())[-8:]
        filename = "./{dirname}/{sequence}_{random_id}.png".format(dirname=self._base_dir_name, sequence=sequence, random_id=random_id)
        self.captcha_generator.write(sequence, filename)
        return filename

    def generate_claptcha_image(self, sequence):
        """
        Receive a CAPTCHA sequence (string), creates a CAPTCHA image, and returns the generated file name.
        Using Calptcha to generate
        """
        random_id = str(uuid.uuid4())[-8:]
        filename = "./{dirname}/{sequence}_{random_id}.png".format(dirname=self._base_dir_name, sequence=sequence, random_id=random_id)
        
        noise_level = round(random.uniform(0.0, 0.1), ndigits=3) # Randomly select a noise level for our generation, from a uniform distribution
        # c = Claptcha(sequence, self.font, (self.img_w / 1.5, self.img_h / 1.5), noise=noise_level, resample=Image.BICUBIC)
        c = Claptcha(sequence, self.font, (self.img_w / 1.5, self.img_h / 1.5), noise=noise_level, resample=Image.NEAREST)
        c.write(filename)
        return filename
    
    def execute_img_generation(self, num_of_images=10):
        """
        Generate N CAPTCHA images, N received as a parameter.
        Each image is saved as a PNG file
        """
        for i in range(num_of_images):
            gen_type = self.select_captcha_generator()
            print("Running generation number #{i} (type = {t})".format(i=i+1, t=gen_type))
            sequence = self.generate_captcha_sequence()

            if gen_type == 1: # PyCaptcha
                self.generate_pycaptcha_image(sequence)
            elif gen_type == 2: # Claptcha
                self.generate_claptcha_image(sequence)

ig = ImageGenerator()
ig.execute_img_generation(num_of_images=20)

# import tarfile
# tar = tarfile.open("local_test_lowercase_ascii.tar.gz", "w:gz")
# tar.add("local_test_lowercase_ascii/", arcname="local_test_lowercase_ascii")
# tar.close()