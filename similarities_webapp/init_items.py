# own modules
from fashion_similarities.utils import GetSimpleData
from fashion_similarities.lsh import LSH

# standard libraries
import numpy as np
import pandas as pd
import sys
import os
import pickle
from tensorflow.keras import losses
from keras import Model
import tensorflow as tf
from argparse import ArgumentParser
import warnings

# django imports
sys.path.append("../")
os.environ['DJANGO_SETTINGS_MODULE'] = 'similarities_webapp.settings'
import django

django.setup()
from polls.models import Item, Image
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.http import Http404

# --------------------------------
parser = ArgumentParser()
parser.add_argument(
    '--kind', '-k', dest='kind', type=str, default="train", help="specify if populate with train or test data"
)
parser.add_argument(
    '--path', '-p', help="specify the path for the data to load", required=True
)
args = parser.parse_args()
kind = args.kind
path = args.path
data_size = 'full'
train = True if kind == "train" else False
hash_size = 9
hash_func = "euclidean"
dist_func = "euclidean"
total_hashes = 1500


# --------------------------------


class DBPopulator:
    def __init__(
            self, data, meta, total_hashes, hash_size, hash_func, dist_func=None,
            image_width=28, image_height=28, channels=3, latent_dim=64
    ):
        self.data = data
        self.meta = meta
        self.embeddings = {}
        self.total_hashes = total_hashes
        self.hash_size = hash_size
        self.hash_func = hash_func
        self.dist_func = dist_func or hash_func  # set dist_func to be the same as hash func if not specified otherwise
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.latent_dim = latent_dim

    def populate_item_model(self):
        objects, images = [], []
        for i, image in enumerate(self.data):
            meta = self.meta.loc[self.meta.idx == i, :]
            if meta.shape[0] == 0:
                warnings.warn(f"no meta data available for image in row {i}. Skipping")
                continue
            id = meta.id.to_list()[0]
            try:
                obj = get_object_or_404(Item, pk=id)
                if kind == "test" and obj.train:
                    obj.delete()
            except Http404:
                objects.append(Item(
                    id=id,
                    train=train,
                    item_gender=meta.gender.to_list()[0],
                    item_type=meta.articleType.to_list()[0],
                    item_color=meta.baseColour.to_list()[0],
                ))
                images.append(Image(
                    item=objects[-1],
                    image=image.tobytes()
                ))
            if i % 1000 == 0:
                print(f"Done processing {i}th image. Bulk write to DB")
                Item.objects.bulk_create(objects)
                Image.objects.bulk_create(images)
                objects, images, exclude = [], [], []
        print(f"Done processing {i}th image. Bulk write to DB")
        Item.objects.bulk_create(objects)
        Image.objects.bulk_create(images)

    def populate_hash_table_model(self):
        def get_embedding(encoder, image_data):
            # get the encoder layer
            layer_name = 'encoded'
            encoder = Model(inputs=encoder.input, outputs=encoder.get_layer(layer_name).output)
            image_data = np.expand_dims(image_data, 0) if not len(image_data.shape) == 4 else image_data
            image_data = image_data.astype('float32') / 255.
            encoded_img = encoder.predict(image_data)
            return encoded_img

        model_path = os.path.join(settings.MODELS, 'autoencoder')
        if os.path.exists(model_path):
            autoencoder = tf.keras.models.load_model(model_path)
        else:
            raise ReferenceError(f"model not found in path {model_path}")
        print("make embeddings...")
        for chunk in np.array_split(self.meta, self.meta.shape[0] // 1000 + 1):
            print("...processing chunk...")
            idx_slices = chunk.idx.tolist()
            images = self.data[idx_slices]
            embs = get_embedding(autoencoder, images)
            embs = {chunk.loc[chunk.idx == idx_slices[i]].id.to_list()[0]: embs[i] for i in range(len(embs))}
            self.embeddings.update(embs)

        print("Done. Building hash tables now")
        lsh = LSH(
            data=self.embeddings, rows_per_band=self.hash_size, num_total_hashes=self.total_hashes,
            hash_func=self.hash_func, dist_metric=self.dist_func, quantile=.4
        )
        lsh.build_hashtables()
        print("dump lsh class for future use")
        path = os.path.join(settings.MODELS, 'lsh.p')
        pickle.dump(lsh, open(path, "wb"))


def main():
    print("Load Data")
    img_path = os.path.join(path, "img_" + kind)  # todo wieder kind eintragen
    metadata_path = os.path.join(path, "styles.csv")
    data, id = GetSimpleData.load_img_from_path(img_path, return_id=True)
    mapper = pd.DataFrame.from_dict(id, orient="index", columns=["id"]).reset_index().rename(columns={"index": "idx"})
    mapper.id = mapper.id.astype("int")
    meta = pd.read_csv(metadata_path, error_bad_lines=False).merge(mapper, right_on="id", left_on="id", how="inner")
    meta = meta.loc[~meta.masterCategory.isin(["Home", "Personal Care", "Free Items"]), :]
    print("Done loading Data. Now initializing objects")
    pop = DBPopulator(data, meta, total_hashes=total_hashes, hash_size=hash_size, hash_func=hash_func,
                      dist_func=dist_func, image_width=224, image_height=224, channels=3)
   # pop.populate_item_model()
    if train:
        pop.populate_hash_table_model()


if __name__ == "__main__":
    main()
