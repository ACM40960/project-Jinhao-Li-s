from genplate import *
import matplotlib.pyplot as plt
# Generate license plate number data for training purposes
class input_data:
    def __init__(self,batch_size , width, height):
        super(input_data, self).__init__()
        self.genplate = GenPlate("./data/font/platech.ttf", './data/font/platechar.ttf', "./data/NoPlates")
        self.batch_size = batch_size
        self.height = height
        self.width = width
    def iter(self):
        data = []
        label = []
        for i in range(self.batch_size):
            img, tag = self.sample(self.genplate, self.width, self.height)
            data.append(img)
            label.append(tag)
        return np.array(data), np.array(label)
    @staticmethod
    def range(lo, hi):
        return lo + r(hi - lo)
    def lab(self):
        name = ""
        label = list([])
        #Generates the first Chinese character label for a license plate
        label.append(self.range(0, 31)) 
        #Generates the 2th letter label for a license plate
        label.append(self.range(41, 65)) 
        #The tag that produces the next five letters of the license plate
        for i in range(5):
            label.append(self.range(31, 65))   
        name += chars[label[0]]
        name += chars[label[1]]
        for i in range(5):
            name += chars[label[i+2]]
        return name, label

    def sample(self, genplate, width, height):
        num, label = self.lab()
        img = genplate.generate(num)
        img = cv.resize(img, (height, width))
        img = np.multiply(img, 1/255.0)
        #The returned 'label' is the label, and 'img' is the license plate image
        return img, label        


