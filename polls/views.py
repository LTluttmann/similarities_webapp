from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django.db.models import Max
from .models import Choice, Item
from .apps import LSHConfig, AutoencoderConfig
import numpy as np
import io
import urllib, base64
from PIL import Image
import sys
sys.path.append("../../fashion_similarities/python/src")
from fashion_similarities.lsh import LSH
import logging
from keras import Model

#logging.basicConfig(filename='example.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --------CONFIG------------  # Todo auslagern

img_width = 224
img_hight = 224
channels = 3
layer_name = 'encoded'
encoder = Model(
    inputs=AutoencoderConfig.autoencoder.input,
    outputs=AutoencoderConfig.autoencoder.get_layer(layer_name).output
)


def get_embedding(encoder, image_data):
    image_data = np.expand_dims(image_data, 0) if not len(image_data.shape) == 4 else image_data
    image_data = image_data.astype('float32') / 255.
    encoded_img = encoder.predict(image_data)
    return encoded_img

# --------------------------


def generate_uri(image):
    query_image = np.frombuffer(image, dtype="uint8").reshape(img_width, img_hight, channels)
    query_image = Image.fromarray(query_image, 'RGB')
    data = io.BytesIO()
    query_image.save(data, "JPEG")  # pick your format
    data64 = base64.b64encode(data.getvalue())
    uri = u'data:img/jpeg;base64,'+data64.decode('utf-8')
    return uri


def index(request):
    return render(request, 'polls/index.html', dict())


def retrieve(request):
    pass


def vote(request):
    try:
        selected_choice = request.POST['choice']
    except KeyError:
        print("getting random item from the test set")
        # query_set = [
        #     16627, 24715, 21774, 16101, 9378, 2136, 11746, 34573, 37726, 32179, 14753, 25284, 18282, 37139, 38979,
        #     35182, 47629, 42056, 26678, 38983, 8366, 21401, 43522, 19853, 10652, 45359, 28941, 8863, 45377, 20888,
        #     12665, 8834, 31440, 34914, 44923, 32401, 6383, 9602, 11057, 29648, 33895, 25451, 42203,
        #     9045, 7935, 33486, 19845, 24532, 7110, 20168, 41676, 39227, 22613, 16159, 27855
        # ]
        # query_item = np.random.choice(query_set)
        query_item = np.random.choice(Item.objects.filter(train=False).values_list("id", flat=True))
        print("query: ", query_item)
        query_item = Item.objects.get(pk=query_item)
        print("Got item. Generating the embedding")
        query_image = query_item.image_set.get().image
        query_image = np.frombuffer(query_image, dtype="uint8").reshape(1, img_width, img_hight, channels)
        embedding = get_embedding(encoder, query_image)

        # use embedding to retrieve a neighbor (similar image)
        print("starting lsh")
        lsh = LSHConfig.data
        k = 4
        items = lsh.get_near_duplicates(
            np.frombuffer(embedding, dtype="float32"), num_duplicates=k, verbose=False
        )
        print(f"LSH retrived {len(items)} items. now load the from database")
        neighbors = Item.objects.filter(pk__in=[items[i][0] for i in range(k)]).values_list("id", flat=True)
        neighbors = [
            item.image_set.get().image for item in Item.objects.filter(pk__in=[items[i][0] for i in range(k)])
        ]
        # as baseline use a image from the same class (e.g. skirt) as the query
        print("get random item")
        random_item = Item.objects.filter(
            train=True,
            item_gender=query_item.item_gender,
            item_type=query_item.item_type,
            item_color=query_item.item_color
        ).values_list("id", flat=True)
        if len(random_item) == 0:
            random_item = Item.objects.filter(
                train=True,
                item_type=query_item.item_type
            ).values_list("id", flat=True)
        random_item_image = Item.objects.get(pk=np.random.choice(random_item)).image_set.get().image
        print("generate uris")
        # generate uris and pass them to the view
        uri_query = generate_uri(query_item.image_set.get().image)
        uri_neighbors = [generate_uri(neighbor) for neighbor in neighbors]
        uri_random = generate_uri(random_item_image)
        uri_neighbors.append(uri_random)
        context = {
            'query': uri_query,
            'neighbors': uri_neighbors,
            'random': uri_random,
            'k_range': list(range(k+1)),
            'k_max': k+1
        }
        print("create the website object")
        return render(request, 'polls/checker.html', context)
    else:
        print(selected_choice)
        c = Choice(choice=selected_choice)
        c.save()
        return HttpResponseRedirect(reverse('polls:vote'))
