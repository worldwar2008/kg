import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.

def to_Games(x):
    # I have gone through 'almost' manually through existing categories and came up with this non-elegant regex
    if re.search('([gG]am)|([pP]oker)|([cC]hess)|([pP]uzz)|([bB]all)|([pP]ursu)|([fF]ight)|([sS]imulat)|([sS]hoot)',
                 x) is not None:
        return 'Games'
    # Then I went through existing abbreviations like RPG, MMO and so on
    if re.search('(RPG)|(SLG)|(RAC)|(MMO)|(MOBA)', x) is not None:
        return 'Games'
    # Still small list of items left which is not covered by regex
    if x in ['billards', 'World of Warcraft', 'Tower Defense', 'Tomb', 'Ninja', 'Europe and Fantasy', 'Senki',
             'Shushan', 'Lottery ticket', 'majiang', 'tennis', 'Martial arts']:
        return 'Games'
    else:
        return x


def to_Property(x):
    # All property/estate stuff will be place into Property category
    if x in ['Property Industry 2.0', 'Property Industry new', 'Property Industry 1.0']:
        return 'Property'
    if re.search('([eE]state)', x) is not None:
        return 'Property'
    else:
        return x


def to_Family(x):
    if re.search(
            '([fF]amili)|([mM]othe)|([fF]athe)|(bab)|([rR]elative)|([pP]regnan)|([pP]arent)|([mM]arriag)|([lL]ove)',
            x) is not None:
        return 'Family'
    else:
        return x


def to_Fun(x):
    '''One can argue about my decision however I used following rules:
       - all comics -> Fun
       - all animation/painting -> Fun
       - all things labeled as trend or passion or community -> Fun
       - all things I could identify as messangers -> Fun
       - all things related to images/pictures -> Fun
       - horoscopes -> Fun
       - jokes -> Fun
       - I don\'t know what is Parkour avoid but it goes to -> Fun'''
    if re.search('([fF]un)|([cC]ool)|([tT]rend)|([cC]omic)|([aA]nima)|([pP]ainti)|\
                 ([fF]iction)|([pP]icture)|(joke)|([hH]oroscope)|([pP]assion)|([sS]tyle)|\
                 ([cC]ozy)|([bB]log)', x) is not None:
        return 'Fun'
    if x in ['Parkour avoid class', 'community', 'Enthusiasm', 'cosplay', 'IM']:
        return 'Fun'
    else:
        return x


def to_Productivity(x):
    if x == 'Personal Effectiveness 1' or x == 'Personal Effectiveness':
        return 'Productivity'
    else:
        return x


def to_Finance(x):
    if re.search(
            '([iI]ncome)|([pP]rofitabil)|([lL]iquid)|([rR]isk)|([bB]ank)|([fF]uture)|([fF]und)|([sS]tock)|([sS]hare)',
            x) is not None:
        return 'Finance'
    if re.search('([fF]inanc)|([pP]ay)|(P2P)|([iI]nsura)|([lL]oan)|([cC]ard)|([mM]etal)|\
                  ([cC]ost)|([wW]ealth)|([bB]roker)|([bB]usiness)|([eE]xchange)', x) is not None:
        return 'Finance'
    if x in ['High Flow', 'Housekeeping', 'Accounting', 'Debit and credit', 'Recipes', 'Heritage Foundation', 'IMF', ]:
        return 'Finance'
    else:
        return x


def to_Religion(x):
    if x == 'And the Church':
        return 'Religion'
    else:
        return x


def to_Services(x):
    if re.search('([sS]ervice)', x) is not None:
        return 'Services'
    else:
        return x


def to_Travel(x):
    if re.search('([aA]viation)|([aA]irlin)|([bB]ooki)|([tT]ravel)|\
                  ([hH]otel)|([tT]rain)|([tT]axi)|([rR]eservati)|([aA]ir)|([aA]irport)', x) is not None:
        return 'Travel'
    if re.search('([jJ]ourne)|([tT]ransport)|([aA]ccommodat)|([nN]avigat)|([tT]ouris)|([fF]light)|([bB]us)',
                 x) is not None:
        return 'Travel'
    if x in ['High mobility', 'Destination Region', 'map', 'Weather', 'Rentals']:
        return 'Travel'
    else:
        return x


def to_Custom(x):
    if re.search('([cC]ustom)', x) is not None:
        return 'Custom'
    else:
        return x


def to_Video(x):
    # not sure if round means Rounds app for group chat, but I stick to this hypothesis. Might be popular app in China
    if x in ['video', 'round', 'the film', 'movie']:
        return 'Video'
    else:
        return x


def to_Shopping(x):
    if x in ['Smart Shopping', 'online malls', 'online shopping by group, like groupon', 'takeaway ordering',
             'online shopping, price comparing', 'Buy class', 'Buy', 'shopping sharing',
             'Smart Shopping 1', 'online shopping navigation']:
        return 'Shopping'
    else:
        return x


def to_Education(x):
    if re.search('([eE]ducati)|([rR]ead)|([sS]cienc)|([bB]ooks)', x) is not None:
        return 'Education'
    if x in ['literature', 'Maternal and child population', 'psychology', 'exams', 'millitary and wars', 'news',
             'foreign language', 'magazine and journal', 'dictionary', 'novels', 'art and culture',
             'Entertainment News',
             'College Students', 'math', 'Western Mythology', 'Technology Information', 'study abroad',
             'Chinese Classical Mythology']:
        return 'Education'
    else:
        return x


def to_Vitality(x):
    if x in ['vitality', '1 vitality']:
        return 'Vitality'
    if x in ['sports and gym', 'Health Management', 'Integrated Living', 'Medical', 'Free exercise', 'A beauty care',
             'fashion', 'fashion outfit', 'lose weight', 'health', 'Skin care applications', 'Wearable Health']:
        return 'Vitality'
    else:
        return x


def to_Sports(x):
    if x in ['sports', 'Sports News']:
        return 'Sports'
    else:
        return x


def to_Music(x):
    if x == 'music':
        return 'Music'
    else:
        return x


def to_Travel_2(x):
    if re.search('([hH]otel)', x) is not None:
        return 'Travel'
    else:
        return x


def to_Other(x):
    if x in ['1 free',
             'The elimination of class',
             'unknown',
             'free',
             'comfortable',
             'Cozy 1',
             'other',
             'Total Cost 1',
             'Classical 1',
             'Quality 1',
             'classical',
             'quality',
             'Car Owners',
             'Noble 1',
             'Pirated content',
             'Securities',
             'professional skills',
             'Jobs',
             'Reputation',
             'Simple 1',
             '1 reputation',
             'Condition of the vehicles',
             'magic',
             'Internet Securities',
             'weibo',
             'Housing Advice',
             'notes',
             'farm',
             'Nature 1',
             'Total Cost',
             'Sea Amoy',
             'show',
             'Car',
             'pet raising up',
             'dotal-lol',
             'Express',
             'radio',
             'Occupational identity',
             'Utilities',
             'Trust',
             'Contacts',
             'Simple',
             'Automotive News',
             'Sale of cars',
             'File Editor',
             'network disk',
             'class managemetn',
             'management',
             'natural',
             'Points Activities',
             'Decoration',
             'store management',
             'Maternal and child supplies',
             'Tour around',
             'coupon',
             'User Community',
             'Vermicelli',
             'noble',
             'poetry',
             'Antique collection',
             'Reviews',
             'Scheduling',
             'Beauty Nail',
             'shows',
             'Hardware Related',
             'Smart Home',
             'Sellers',
             'Desktop Enhancements',
             'library',
             'entertainment',
             'Calendar',
             'Ping',
             'System Tools',
             'KTV',
             'Behalf of the drive',
             'household products',
             'Information',
             'Man playing favorites',
             'App Store',
             'Engineering Drawing',
             'Academic Information',
             'Appliances',
             'Peace - Search',
             'Make-up application',
             'WIFI',
             'phone',
             'Doctors',
             'Smart Appliances',
             'reality show',
             'Harem',
             'trickery',
             'Jin Yong',
             'effort',
             'Xian Xia',
             'Romance',
             'tribe',
             'email',
             'mesasge',
             'Editor',
             'Clock',
             'search',
             'Intelligent hardware',
             'Browser',
             'Furniture']:
        return 'Other'
    else:
        return x

def get_categ():
    labels = pd.read_csv("../input/label_categories.csv")
    app_labels = pd.read_csv("../input/app_labels.csv")
    apps = pd.merge(app_labels, labels, how='left', on='label_id')
    # print apps.shape
    # print apps['category'].value_counts()
    apps['general_groups'] = apps['category']
    apps['Games'] = apps['general_groups'].apply(to_Games)
    apps['Property'] = apps['general_groups'].apply(to_Property)
    apps['Family'] = apps['general_groups'].apply(to_Family)
    apps['Fun'] = apps['general_groups'].apply(to_Fun)
    apps['Productivity'] = apps['general_groups'].apply(to_Productivity)
    apps['Finance'] = apps['general_groups'].apply(to_Finance)
    apps['Religion'] = apps['general_groups'].apply(to_Religion)
    apps['Services'] = apps['general_groups'].apply(to_Services)
    apps['Travel'] = apps['general_groups'].apply(to_Travel)
    apps['Custom'] = apps['general_groups'].apply(to_Custom)
    apps['Video'] = apps['general_groups'].apply(to_Video)
    apps['Shopping'] = apps['general_groups'].apply(to_Shopping)
    apps['Education'] = apps['general_groups'].apply(to_Education)
    apps['Vitality'] = apps['general_groups'].apply(to_Vitality)
    apps['Sports'] = apps['general_groups'].apply(to_Sports)
    apps['Music'] = apps['general_groups'].apply(to_Music)
    apps['Travel_2'] = apps['general_groups'].apply(to_Travel_2)
    apps['Other'] = apps['general_groups'].apply(to_Other)
    return apps




