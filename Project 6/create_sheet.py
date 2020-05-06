import numpy as np
import csv

def main():

    with open('train_all.csv', mode='w+') as train_file:
        writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        i = 0
        class_label = 0

        for i in range(0,25000):

            if i>12499:
                class_label = 1

            writer.writerow(['{}.png'.format(i),'{}'.format(class_label)])

if __name__ == '__main__':
    main()