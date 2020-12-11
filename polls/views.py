from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.db.models import Max
from .models import Choice, Question, Item
from .apps import LSHConfig, AutoencoderConfig
import numpy as np
import io
import urllib, base64
from PIL import Image
import sys
sys.path.append("../../fashion_similarities/python/src")
from fashion_similarities.lsh import LSH
import logging

logging.basicConfig(filename='example.log', level=logging.DEBUG)

def generate_uri(image):
    query_image = np.frombuffer(image, dtype="uint8").reshape(28, 28)
    query_image = Image.fromarray(query_image, 'L')
    data = io.BytesIO()
    query_image.save(data, "JPEG")  # pick your format
    data64 = base64.b64encode(data.getvalue())
    uri = u'data:img/jpeg;base64,'+data64.decode('utf-8')
    return uri


def index(request):
    return render(request, 'polls/index.html', dict())


def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/detail.html', {'question': question})


def results(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    return render(request, 'polls/results.html', {'question': question})


def check_if_same(lsh_class: LSH):
    i = np.random.choice(np.arange(0, lsh_class.data.shape[0]))
    lsh_data = lsh_class.data[i]
    item_data = Item.objects.get(pk=i+1)
    item_data2 = Item.objects.get(pk=i-1)
    item_data = np.frombuffer(item_data.embedding, dtype="float32")
    item_data2 = np.frombuffer(item_data2.embedding, dtype="float32")
    print("django: \n ", item_data)
    print("django: \n ", item_data2)
    print("Lsh: \n", lsh_data)


def vote(request):
    try:
        selected_choice = request.POST['choice']
    except KeyError:
        query_item = np.random.choice(Item.objects.filter(train=False))

        # generate embedding if not yet there
        if not query_item.embedding:
            query_image = np.frombuffer(query_item.image, dtype="uint8").reshape(1, 28, 28)
            query_image = query_image.astype('float32') / 255.
            logging.info(query_image.shape)
            logging.info(query_image)
            autoencoder = AutoencoderConfig.autoencoder
            embedding = autoencoder.encoder(query_image)
            query_item.embedding = embedding.numpy().tobytes()
            query_item.save()

        # use embedding to retrieve a neighbor (similar image)
        lsh = LSHConfig.data
        neighbor = Item.objects.get(
            pk=lsh.get_near_duplicates(np.frombuffer(query_item.embedding, dtype="float32"), num_duplicates=1)[0][0]
        )

        # as baseline use a image from the same class (e.g. skirt) as the query
        random_item = np.random.choice(
            Item.objects.filter(item_class=query_item.item_class)  # filter(item_class=query_item.item_class)
        )

        # generate uris and pass them to the view
        uri_query = generate_uri(query_item.image)
        uri_neighbor = generate_uri(neighbor.image)
        uri_random = generate_uri(random_item.image)
        context = {
            'query': uri_query,
            'neighbor': uri_neighbor,
            'random': uri_random
        }
        return render(request, 'polls/checker.html', context)
    else:
        print(selected_choice)
        c = Choice(choice=selected_choice)
        c.save()
        return HttpResponseRedirect(reverse('polls:vote'))
    # try:
    #     selected_choice = question.choice_set.get(pk=request.POST['choice'])
    # except (KeyError, Choice.DoesNotExist):
    #     # Redisplay the question voting form.
    #     return render(request, 'polls/detail.html', {
    #         'question': question,
    #         'error_message': "You didn't select a choice.",
    #     })
    # else:
    #     selected_choice.votes += 1
    #     selected_choice.save()
    #     # Always return an HttpResponseRedirect after successfully dealing
    #     # with POST data. This prevents data from being posted twice if a
    #     # user hits the Back button.
    #     return HttpResponseRedirect(reverse('polls:results', args=(question.id,)))
