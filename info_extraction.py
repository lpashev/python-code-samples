from nltk import pos_tag, word_tokenize
from os import listdir
from nltk.corpus import gazetteers
from os.path import isfile, join
from gensim.models import Word2Vec
import nltk
import os
import re


dir = os.getcwd()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('gazetteers')

mytrainingpath = dir + '/training'
mytestpath = dir + '/testdata'
taggedpath = dir + '/testdata_tagged'
categorizedpath = dir + '/categorized'

model = Word2Vec.load_word2vec_format('/home/projects/google-news-corpus/GoogleNews-vectors-negative300.bin',
                                      binary=True)

trainingfiles = [f for f in listdir(mytrainingpath) if isfile(join(mytrainingpath, f))]
testfiles = [f for f in listdir(mytestpath) if isfile(join(mytestpath, f))]
taggedfiles = [f for f in listdir(taggedpath) if isfile(join(taggedpath, f))]

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
speakers = []
locations = []


def loadData(files, path):
    data = []
    for f in files:
        data.append(gazetteers.words(path + '/' + f))
    return data


def isNonSent(sent):
    return 'SPEAKER:' in sent or 'Appointment:' in sent or 'Host:' in sent or 'PostedBy:' in sent or sent.isupper() or 'Abstract:' in sent or 'TITLE:' in sent or '   ' in sent or 'Dates:' in sent or 'WHEN:' in sent or 'Type:' in sent or 'Who:' in sent or 'Topic:' in sent or 'HOST:' in sent or 'WHERE:' in sent or 'Where:' in sent or 'Time:' in sent or 'Place:' in sent or '--' in sent or 'WHO:' in sent or 'TIME:' in sent or 'PLACE:' in sent


def findTime(string):
    startIndex = (string.index('Time:'))
    endIndex = string[startIndex + 5:].index('\n') + 5 + startIndex
    time = string[startIndex + 5:endIndex].strip()
    return time


def findPlace(string):
    if 'Place:' in string:
        startIndex = (string.index('Place:'))
    if 'Where:' in string:
        startIndex = (string.index('Where:'))
    if 'WHERE:' in string:
        startIndex = (string.index('WHERE:'))
    if 'PLACE:' in string:
        startIndex = (string.index('PLACE:'))
    try:
        endIndex = string[startIndex + 6:].index('\n') + 6 + startIndex
    except ValueError:
        endIndex = len(string) - 1
    place = string[startIndex + 6:endIndex].strip()
    return place


def findTopic(string):
    if 'Topic:' in string:
        startIndex = (string.index('Topic:'))
    endIndex = string[startIndex + 6:].index('\n') + 6 + startIndex
    topic = string[startIndex + 6:endIndex].strip()
    return topic


def fineSpeaker(string):
    if 'Speaker:' in string:
        startIndex = (string.index('Speaker:'))
        try:
            endIndex = string[startIndex + 8:].index('\n') + 8 + startIndex
        except ValueError:
            endIndex = len(string) - 1
        speaker = string[startIndex + 8:endIndex].strip()
    if 'SPEAKER:' in string:
        startIndex = (string.index('SPEAKER:'))
        try:
            endIndex = string[startIndex + 8:].index('\n') + 8 + startIndex
        except ValueError:
            endIndex = len(string) - 1
        speaker = string[startIndex + 8:endIndex].strip()
    if 'Who:' in string:
        startIndex = (string.index('Who:'))
        try:
            endIndex = string[startIndex + 4:].index('\n') + 4 + startIndex
        except ValueError:
            endIndex = len(string) - 1
        speaker = string[startIndex + 4:endIndex].strip()
    if 'WHO:' in string:
        startIndex = (string.index('WHO:'))
        try:
            endIndex = string[startIndex + 4:].index('\n') + 4 + startIndex
        except ValueError:
            endIndex = len(string) - 1
        speaker = string[startIndex + 4:endIndex].strip()
    return speaker


def paragraphSents(sents):
    newsents = []
    flag = False
    lastsent = ''
    for sent in sents:
        currentsent = sent
        if (not isNonSent(sent) and flag) or (isNonSent(lastsent) and not isNonSent(sent)):
            currentsent = '<pararaph>' + currentsent
            flag = False
        if (sent.endswith('.') or sent.endswith('. ')) and (not isNonSent(sent)):
            currentsent = currentsent + '</paragraph>'
            flag = True
        lastsent = sent
        currentsent = re.sub(r'\. +', '. ', currentsent)
        newsents.append(currentsent)
    return newsents


# Load training set
training_sents = loadData(trainingfiles, mytrainingpath)
# Load test set
test_sents = loadData(testfiles, mytestpath)
# Load pretagged set
taggedsents = loadData(taggedfiles, taggedpath)

# Extract data from training set
for text in training_sents:
    text = '\n'.join(text)

    speaker_search = re.findall(r'<speaker>\w+.\w+</speaker>', text, re.S)
    for reg in speaker_search:
        if reg:
            speakers.append(re.sub(r'</speaker>', "", (re.sub(r'<speaker>', '', reg))))
    locations_search = re.findall(r'<location>.{1,44}</location>', text, re.S)
    for reg in locations_search:
        if reg:
            locations.append(re.sub(r'</location>', "", (re.sub(r'<location>', '', reg))))
speakers = list(set(speakers))
locations = list(set(locations))

newfiles = []
for file in test_sents:
    newsents = []
    for sent in tokenizer.tokenize('\n'.join([f for f in paragraphSents(file)])):
        if not isNonSent(sent):
            sent = '<sentence>' + sent + '</sentence>'
            sent = re.sub(r'<sentence><paragraph>', '<paragraph><sentence>', sent)
            sent = re.sub(r'</paragraph></sentence>', '</sentence></paragraph>', sent)

        newsents.append(sent)
    newfile = '\n'.join(newsents)
    newfiles.append(newfile)
    newfile = ''

count = 301
systaggedfiles = []
for file in newfiles:
    if 'Time:' in file:
        time = findTime(file)
        timelower = time.lower()
        if time.count(':') > 1:
            (start, end) = time.split('-')
            start = '<stime>' + start + '</stime>'
            end = '<etime>' + end + '</etime>'
            file = re.sub(r'' + time, start + '-' + end, file)
            file = re.sub(r'' + timelower, start + '-' + end, file, re.I)
        if time.count(':') == 1:
            file = re.sub(r'' + time, '<stime>' + time + '</stime>', file)
            file = re.sub(r'' + timelower, '<stime>' + timelower + '</stime>', file, re.I)
    if 'Place:' in file or 'WHERE:' in file or 'PLACE:' in file or 'Where:' in file:
        place = findPlace(file)
    else:
        for loc in locations:
            if loc in file:
                file = file.replace(loc, '<location>' + loc + '</location>')
        file = file.replace(place, '<location>' + place + '</location>')
    if 'Speaker:' in file or 'Who:' in file or 'WHO:' in file or 'SPEAKER:' in file:
        speaker = fineSpeaker(file)
        if ',' in speaker:
            speaker = speaker.split(',')[0]
        else:
            if '-' in speaker:
                speaker = speaker.split('-')[0]
        file = re.sub(r'' + speaker, '<speaker>' + speaker + '</speaker>', file)
    else:
        for spkr in speakers:
            if spkr in file:
                file = re.sub(r'' + spkr, '<speaker>' + spkr + '</speaker>', file)

    f = open(taggedpath + '/' + str(count) + '.txt', "w")
    f.write(file)
    f.close()
    systaggedfiles.append(file)
    count += 1

chars = '*./-_=+^><%[]1234567890@'
topics = ['biology', 'chemistry', 'business', 'mathematics', 'engineering', 'technology', 'physics', 'art', 'law',
          'psychology', 'geography', 'history']
filenouns = []
for file in systaggedfiles:

    file = [f.replace('<location>', '').replace('</location>', '').replace('<speaker>', '').replace('</speaker>',
                                                                                                    '').replace(
        '<paragraph>', '').replace('</paragraph>', '').replace('<stime>', '').replace('</stime>', '').replace(
        '<sentence>', '').replace('</sentence>', '').replace('<etime>', '').replace('</etime>', '') for f in
            file.split('\n')]
    if 'Topic:' in file:
        topic = findTopic(file)
    else:
        topic = ''

    file = [f for f in file if not isNonSent(f) and 'andrew' not in f]
    file = pos_tag(word_tokenize('\n'.join(file) + '\n' + topic))

    nouns = [f[0].lower() for f in file if
             (f[1] == 'NN' or f[1] == 'NNP' or f[1] == 'NNS') and not any((c in chars) for c in f[0])]
    nouns = list(set(nouns))
    filenouns.append(nouns)

categorized = []
for (file, nouns) in zip(systaggedfiles, filenouns):
    maxval = -1
    for noun in nouns:
        for topic in topics:
            try:
                newsim = model.similarity(noun, topic)
            except KeyError:
                newsim = -1
            if newsim > maxval:
                maxval = newsim
                maxtopic = topic

    categorized.append((maxtopic, file))

catcount = 301
for (topic, file) in categorized:
    categs = open(categorizedpath + '/' + str(catcount) + '.txt', "w")
    if topic == 'technology':
        categs.write('COMPUTER SCIENCE\n' + '\n'.join(file))
    else:
        categs.write(topic.upper() + '\n' + '\n'.join(file))
    categs.close()
    catcount += 1
