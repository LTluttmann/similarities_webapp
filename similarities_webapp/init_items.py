# own modules
from fashion_similarities.utils import GetSimpleData
from fashion_similarities.autoencoder import Autoencoder
from fashion_similarities.lsh import LSH

# standard libraries
import numpy as np
import sys
import os
import pickle
from tensorflow.keras import losses
import tensorflow as tf
from argparse import ArgumentParser

# django imports
sys.path.append("../")
os.environ['DJANGO_SETTINGS_MODULE'] = 'similarities_webapp.settings'
import django
django.setup()
from polls.models import Item, HashTable
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Max

# --------------------------------
parser = ArgumentParser()
parser.add_argument(
    '--kind', '-k', dest='kind', type=str, default="train", help="specify if populate with train or test data"
)
args = parser.parse_args()
kind = args.kind
data_size = 'full'
train = True if kind == "train" else False
hash_size = 10
hash_func = "cosine"
dist_func = "euclidean"
total_hashes = 500
# --------------------------------


class DBPopulator:
    def __init__(
            self, data, classes, total_hashes, hash_size, hash_func, dist_func=None, embeddings=None,
            image_width=28, image_height=28, latent_dim=64
    ):
        self.data = data
        self.classes = classes
        self.embeddings = embeddings
        self.total_hashes = total_hashes
        self.hash_size = hash_size
        self.hash_func = hash_func
        self.dist_func = dist_func or hash_func  # set dist_func to be the same as hash func if not specified otherwise
        self.image_width = image_width
        self.image_height = image_height
        self.latent_dim = latent_dim

    def make_embeddings(self):
        data = self.data.reshape(self.data.shape[0], self.image_width, self.image_height)
        data = data.astype('float32') / 255.

        model_path = os.path.join(settings.MODELS, 'autoencoder')

        if os.path.exists(model_path):
            autoencoder = tf.keras.models.load_model(model_path)
        else:
            autoencoder = Autoencoder(self.latent_dim)
            autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
            autoencoder.fit(data, data,
                            epochs=10,
                            shuffle=True,
                            validation_data=(data, data))
            tf.keras.models.save_model(autoencoder, model_path)
        # do the actual encoding
        encoded_imgs = autoencoder.encoder(data).numpy()
        return encoded_imgs

    def populate_item_model(self):
        if train:
            self.embeddings = self.make_embeddings()  # todo how to import model specifications?
        ids = list(Item.objects.values_list('id', flat=True))
        if len(ids) == self.data.shape[0]:
            return
        print("Done loading Data. Now initializing objects")
        objects = []
        exclude = []
        start_idx = list(Item.objects.aggregate(Max('id')).values())[0] or -1
        for i, image in enumerate(self.data):
            idx = start_idx + i + 1
            try:
                exclude.append(get_object_or_404(Item, pk=idx))
            except:
                objects.append(Item(
                    id=idx,
                    image=image.tobytes(),
                    embedding=self.embeddings[i].tobytes() if train else None,
                    train=train,
                    item_class=self.classes[i]
                ))
        print("done initializing objects. Now bulk write to DB")
        objects = list(set(objects).difference(exclude))
        Item.objects.bulk_create(objects)

    def populate_hash_table_model(self):
        path = os.path.join(settings.MODELS, 'lsh.p')
        # if os.path.exists(path):
        #     with open(path, 'rb') as pickled:
        #         lsh = pickle.load(pickled)
        # else:
        print("Building hash tables now")
        lsh = LSH(
            self.embeddings, rows_per_band=self.hash_size, num_total_hashes=self.total_hashes, hash_func=self.hash_func,
            dist_metric=self.dist_func, bucket_width=10
        )
        lsh.build_hashtables()
        print("dump lsh class for future use")
        pickle.dump(lsh, open(path, "wb"))
        # print("Done building hash tables. Now initialize objects")
        # objects = []
        # for table_idx, table in enumerate(lsh.hash_tables):
        #     for hash_key, items in table.items():
        #         for item in items:
        #             h = HashTable(
        #                 table_id=table_idx,
        #                 hash_key=hash_key,
        #                 item_id=Item.objects.get(pk=item)
        #             )
        #             objects.append(h)
        # print("done initializing objects. Now bulk write to DB")
        # HashTable.objects.bulk_create(objects)


if __name__ == "__main__":
    print("Load Data")
    x, y = GetSimpleData.load_from_github(kind=kind)
    pop = DBPopulator(x, y, total_hashes=500, hash_size=13, hash_func="cosine", dist_func="euclidean", embeddings=None)
    pop.populate_item_model()
    if train:
        pop.populate_hash_table_model()
