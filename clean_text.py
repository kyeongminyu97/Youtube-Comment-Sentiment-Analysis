"""
Functions for Clean/processing text for NLP
"""
import logging
import re
import sys
from unicodedata import category

import pandas as pd
from emoji import UNICODE_EMOJI, demojize, emojize
from ftfy import fix_text

from bs4 import BeautifulSoup
from collections import Counter

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
import string
pd.options.mode.chained_assignment = None

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

from bs4 import BeautifulSoup

def remove_html(text):
    return BeautifulSoup(text, "lxml").text

def remove_punctuation(text):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = string.punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_freqwords(text):
    """custom function to remove the frequent words"""
    cnt = Counter()
    for text in df.values:
        for word in text.split():
            cnt[word] += 1
    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

def remove_rarewords(text):
    """custom function to remove the rare words"""
    n_rare_words = 10
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
    return " ".join([word for word in str(text).split() if word not in RAREWORDS])


def lemmatize_words(text):
    lemmatizer = WordNetLemmatizer()
    wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_emoticons(text):
    EMOTICONS = {
        u":‑\)":"Happy face or smiley",
        u":\)":"Happy face or smiley",
        u":-\]":"Happy face or smiley",
        u":\]":"Happy face or smiley",
        u":-3":"Happy face smiley",
        u":3":"Happy face smiley",
        u":->":"Happy face smiley",
        u":>":"Happy face smiley",
        u"8-\)":"Happy face smiley",
        u":o\)":"Happy face smiley",
        u":-\}":"Happy face smiley",
        u":\}":"Happy face smiley",
        u":-\)":"Happy face smiley",
        u":c\)":"Happy face smiley",
        u":\^\)":"Happy face smiley",
        u"=\]":"Happy face smiley",
        u"=\)":"Happy face smiley",
        u":‑D":"Laughing, big grin or laugh with glasses",
        u":D":"Laughing, big grin or laugh with glasses",
        u"8‑D":"Laughing, big grin or laugh with glasses",
        u"8D":"Laughing, big grin or laugh with glasses",
        u"X‑D":"Laughing, big grin or laugh with glasses",
        u"XD":"Laughing, big grin or laugh with glasses",
        u"=D":"Laughing, big grin or laugh with glasses",
        u"=3":"Laughing, big grin or laugh with glasses",
        u"B\^D":"Laughing, big grin or laugh with glasses",
        u":-\)\)":"Very happy",
        u":‑\(":"Frown, sad, andry or pouting",
        u":-\(":"Frown, sad, andry or pouting",
        u":\(":"Frown, sad, andry or pouting",
        u":‑c":"Frown, sad, andry or pouting",
        u":c":"Frown, sad, andry or pouting",
        u":‑<":"Frown, sad, andry or pouting",
        u":<":"Frown, sad, andry or pouting",
        u":‑\[":"Frown, sad, andry or pouting",
        u":\[":"Frown, sad, andry or pouting",
        u":-\|\|":"Frown, sad, andry or pouting",
        u">:\[":"Frown, sad, andry or pouting",
        u":\{":"Frown, sad, andry or pouting",
        u":@":"Frown, sad, andry or pouting",
        u">:\(":"Frown, sad, andry or pouting",
        u":'‑\(":"Crying",
        u":'\(":"Crying",
        u":'‑\)":"Tears of happiness",
        u":'\)":"Tears of happiness",
        u"D‑':":"Horror",
        u"D:<":"Disgust",
        u"D:":"Sadness",
        u"D8":"Great dismay",
        u"D;":"Great dismay",
        u"D=":"Great dismay",
        u"DX":"Great dismay",
        u":‑O":"Surprise",
        u":O":"Surprise",
        u":‑o":"Surprise",
        u":o":"Surprise",
        u":-0":"Shock",
        u"8‑0":"Yawn",
        u">:O":"Yawn",
        u":-\*":"Kiss",
        u":\*":"Kiss",
        u":X":"Kiss",
        u";‑\)":"Wink or smirk",
        u";\)":"Wink or smirk",
        u"\*-\)":"Wink or smirk",
        u"\*\)":"Wink or smirk",
        u";‑\]":"Wink or smirk",
        u";\]":"Wink or smirk",
        u";\^\)":"Wink or smirk",
        u":‑,":"Wink or smirk",
        u";D":"Wink or smirk",
        u":‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"X‑P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"XP":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":Þ":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"d:":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"=p":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u">:P":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u":‑/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":-[.]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u">:/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=/":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=[(\\\)]":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":L":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u"=L":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":S":"Skeptical, annoyed, undecided, uneasy or hesitant",
        u":‑\|":"Straight face",
        u":\|":"Straight face",
        u":$":"Embarrassed or blushing",
        u":‑x":"Sealed lips or wearing braces or tongue-tied",
        u":x":"Sealed lips or wearing braces or tongue-tied",
        u":‑#":"Sealed lips or wearing braces or tongue-tied",
        u":#":"Sealed lips or wearing braces or tongue-tied",
        u":‑&":"Sealed lips or wearing braces or tongue-tied",
        u":&":"Sealed lips or wearing braces or tongue-tied",
        u"O:‑\)":"Angel, saint or innocent",
        u"O:\)":"Angel, saint or innocent",
        u"0:‑3":"Angel, saint or innocent",
        u"0:3":"Angel, saint or innocent",
        u"0:‑\)":"Angel, saint or innocent",
        u"0:\)":"Angel, saint or innocent",
        u":‑b":"Tongue sticking out, cheeky, playful or blowing a raspberry",
        u"0;\^\)":"Angel, saint or innocent",
        u">:‑\)":"Evil or devilish",
        u">:\)":"Evil or devilish",
        u"\}:‑\)":"Evil or devilish",
        u"\}:\)":"Evil or devilish",
        u"3:‑\)":"Evil or devilish",
        u"3:\)":"Evil or devilish",
        u">;\)":"Evil or devilish",
        u"\|;‑\)":"Cool",
        u"\|‑O":"Bored",
        u":‑J":"Tongue-in-cheek",
        u"#‑\)":"Party all night",
        u"%‑\)":"Drunk or confused",
        u"%\)":"Drunk or confused",
        u":-###..":"Being sick",
        u":###..":"Being sick",
        u"<:‑\|":"Dump",
        u"\(>_<\)":"Troubled",
        u"\(>_<\)>":"Troubled",
        u"\(';'\)":"Baby",
        u"\(\^\^>``":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(\^_\^;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(-_-;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(~_~;\) \(・\.・;\)":"Nervous or Embarrassed or Troubled or Shy or Sweat drop",
        u"\(-_-\)zzz":"Sleeping",
        u"\(\^_-\)":"Wink",
        u"\(\(\+_\+\)\)":"Confused",
        u"\(\+o\+\)":"Confused",
        u"\(o\|o\)":"Ultraman",
        u"\^_\^":"Joyful",
        u"\(\^_\^\)/":"Joyful",
        u"\(\^O\^\)／":"Joyful",
        u"\(\^o\^\)／":"Joyful",
        u"\(__\)":"Kowtow as a sign of respect, or dogeza for apology",
        u"_\(\._\.\)_":"Kowtow as a sign of respect, or dogeza for apology",
        u"<\(_ _\)>":"Kowtow as a sign of respect, or dogeza for apology",
        u"<m\(__\)m>":"Kowtow as a sign of respect, or dogeza for apology",
        u"m\(__\)m":"Kowtow as a sign of respect, or dogeza for apology",
        u"m\(_ _\)m":"Kowtow as a sign of respect, or dogeza for apology",
        u"\('_'\)":"Sad or Crying",
        u"\(/_;\)":"Sad or Crying",
        u"\(T_T\) \(;_;\)":"Sad or Crying",
        u"\(;_;":"Sad of Crying",
        u"\(;_:\)":"Sad or Crying",
        u"\(;O;\)":"Sad or Crying",
        u"\(:_;\)":"Sad or Crying",
        u"\(ToT\)":"Sad or Crying",
        u";_;":"Sad or Crying",
        u";-;":"Sad or Crying",
        u";n;":"Sad or Crying",
        u";;":"Sad or Crying",
        u"Q\.Q":"Sad or Crying",
        u"T\.T":"Sad or Crying",
        u"QQ":"Sad or Crying",
        u"Q_Q":"Sad or Crying",
        u"\(-\.-\)":"Shame",
        u"\(-_-\)":"Shame",
        u"\(一一\)":"Shame",
        u"\(；一_一\)":"Shame",
        u"\(=_=\)":"Tired",
        u"\(=\^\·\^=\)":"cat",
        u"\(=\^\·\·\^=\)":"cat",
        u"=_\^=	":"cat",
        u"\(\.\.\)":"Looking down",
        u"\(\._\.\)":"Looking down",
        u"\^m\^":"Giggling with hand covering mouth",
        u"\(\・\・?":"Confusion",
        u"\(?_?\)":"Confusion",
        u">\^_\^<":"Normal Laugh",
        u"<\^!\^>":"Normal Laugh",
        u"\^/\^":"Normal Laugh",
        u"\（\*\^_\^\*）" :"Normal Laugh",
        u"\(\^<\^\) \(\^\.\^\)":"Normal Laugh",
        u"\(^\^\)":"Normal Laugh",
        u"\(\^\.\^\)":"Normal Laugh",
        u"\(\^_\^\.\)":"Normal Laugh",
        u"\(\^_\^\)":"Normal Laugh",
        u"\(\^\^\)":"Normal Laugh",
        u"\(\^J\^\)":"Normal Laugh",
        u"\(\*\^\.\^\*\)":"Normal Laugh",
        u"\(\^—\^\）":"Normal Laugh",
        u"\(#\^\.\^#\)":"Normal Laugh",
        u"\（\^—\^\）":"Waving",
        u"\(;_;\)/~~~":"Waving",
        u"\(\^\.\^\)/~~~":"Waving",
        u"\(-_-\)/~~~ \($\·\·\)/~~~":"Waving",
        u"\(T_T\)/~~~":"Waving",
        u"\(ToT\)/~~~":"Waving",
        u"\(\*\^0\^\*\)":"Excited",
        u"\(\*_\*\)":"Amazed",
        u"\(\*_\*;":"Amazed",
        u"\(\+_\+\) \(@_@\)":"Amazed",
        u"\(\*\^\^\)v":"Laughing,Cheerful",
        u"\(\^_\^\)v":"Laughing,Cheerful",
        u"\(\(d[-_-]b\)\)":"Headphones,Listening to music",
        u'\(-"-\)':"Worried",
        u"\(ーー;\)":"Worried",
        u"\(\^0_0\^\)":"Eyeglasses",
        u"\(\＾ｖ\＾\)":"Happy",
        u"\(\＾ｕ\＾\)":"Happy",
        u"\(\^\)o\(\^\)":"Happy",
        u"\(\^O\^\)":"Happy",
        u"\(\^o\^\)":"Happy",
        u"\)\^o\^\(":"Happy",
        u":O o_O":"Surprised",
        u"o_0":"Surprised",
        u"o\.O":"Surpised",
        u"\(o\.o\)":"Surprised",
        u"oO":"Surprised",
        u"\(\*￣m￣\)":"Dissatisfied",
        u"\(‘A`\)":"Snubbed or Deflated"}
    emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
    return emoticon_pattern.sub(r'', text)

def chat_words_conversion(text):
    chat_words_str = """
    AFAIK=As Far As I Know
    AFK=Away From Keyboard
    ASAP=As Soon As Possible
    ATK=At The Keyboard
    ATM=At The Moment
    A3=Anytime, Anywhere, Anyplace
    BAK=Back At Keyboard
    BBL=Be Back Later
    BBS=Be Back Soon
    BFN=Bye For Now
    B4N=Bye For Now
    BRB=Be Right Back
    BRT=Be Right There
    BTW=By The Way
    B4=Before
    B4N=Bye For Now
    CU=See You
    CUL8R=See You Later
    CYA=See You
    FAQ=Frequently Asked Questions
    FC=Fingers Crossed
    FWIW=For What It's Worth
    FYI=For Your Information
    GAL=Get A Life
    GG=Good Game
    GN=Good Night
    GMTA=Great Minds Think Alike
    GR8=Great!
    G9=Genius
    IC=I See
    ICQ=I Seek you (also a chat program)
    ILU=ILU: I Love You
    IMHO=In My Honest/Humble Opinion
    IMO=In My Opinion
    IOW=In Other Words
    IRL=In Real Life
    KISS=Keep It Simple, Stupid
    LDR=Long Distance Relationship
    LMAO=Laugh My A.. Off
    LOL=Laughing Out Loud
    LTNS=Long Time No See
    L8R=Later
    MTE=My Thoughts Exactly
    M8=Mate
    NRN=No Reply Necessary
    OIC=Oh I See
    PITA=Pain In The A..
    PRT=Party
    PRW=Parents Are Watching
    ROFL=Rolling On The Floor Laughing
    ROFLOL=Rolling On The Floor Laughing Out Loud
    ROTFLMAO=Rolling On The Floor Laughing My A.. Off
    SK8=Skate
    STATS=Your sex and age
    ASL=Age, Sex, Location
    THX=Thank You
    TTFN=Ta-Ta For Now!
    TTYL=Talk To You Later
    U=You
    U2=You Too
    U4E=Yours For Ever
    WB=Welcome Back
    WTF=What The F...
    WTG=Way To Go!
    WUF=Where Are You From?
    W8=Wait...
    7K=Sick:-D Laugher
    """
    chat_words_map_dict = {}
    chat_words_list = []
    for line in chat_words_str.split("\n"):
        if line != "":
            cw = line.split("=")[0]
            cw_expanded = line.split("=")[1]
            chat_words_list.append(cw)
            chat_words_map_dict[cw] = cw_expanded
    chat_words_list = set(chat_words_list)
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
# https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py

