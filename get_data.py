import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from random import randint


def get_data(path):
    minf = 1500
    maxf = 2300
    data = pd.DataFrame()
    scaler = MinMaxScaler(feature_range=(0, 1))
    # n_train = 10000
    # batch_size = 12288
    batch_size = 1024*3
    n_batch = 30 * batch_size     #this should be like n times timesteps * batch_size

    rand_start = randint(1, 70000)

    for root, dirs, files in os.walk(path):
        """
        not sure if dirs sorting is needed
        dirs.sort()
        """

        #sort file names
        files.sort(key=lambda x: x.lower())
        for file in files:
            name = os.path.join(root, file)
            if 'input' in name:
                input = pd.read_table(name, header=None)

                input = input[rand_start:rand_start + 3 * n_batch]   #take 3 parts randomly

                values = input.values
                values = values.astype('float32')

                # normalize

                scaled_x = scaler.fit_transform(values)
                input = pd.DataFrame(scaled_x, columns=['input1', 'input2'])
                # normalize f
                input['f'] = (int(root.strip().split('_')[1]) - minf) / (maxf - minf)

            elif 'output' in name:
                output = pd.read_table(name, header=None)

                output = output[rand_start:rand_start + 3 * n_batch]

                values = output.values
                values = values.astype('float32')

                # no normalize
                # scaled_y = scaler.fit_transform(values)
                output = pd.DataFrame(values, columns=['output1', 'output2'])

                df = pd.concat([input, output], axis=1)
                data = pd.concat([data, df], sort=False)
            else:
                pass

    return data, batch_size, n_batch



