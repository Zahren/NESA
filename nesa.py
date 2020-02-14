import json
import time
from collections import Counter, defaultdict
from operator import add
from traceback  import print_exc

import nltk
from stanfordcorenlp import StanfordCoreNLP
from google.cloud import bigquery

from engineabcs import RecognizerABC, AnalyzerABC

class GClient:
    """Wrapper for the GC Client"""
    
    def __init__(self, json_name):
        self.client = bigquery.Client.from_service_account_json(json_name)
        
class Loader:
    """Loads the text based data for use in downstream processes."""
    
    def __init__(self, subreddit_name, year, month, amount, client):
	
        self.subreddit_name = subreddit_name
        self.year = str(year)
        if int(month) < 10:
            month = '0' + str(int(month))
        self.month = month
        self.amount = str(int(amount))
        self.client = client
		
        # Run a query on bigquery to retrieve the comments
        self.query()
    
    def query(self):
        query = (
            'SELECT subreddit, body FROM `fh-bigquery.reddit_comments.' + self.year + '_' + self.month + '` '
            'WHERE subreddit = "' + self.subreddit_name + '" '
            'LIMIT '  +  str(self.amount))
        
        self.results = self.client.query(
            query,
            location='US')
        
    def output_list(self):
        comments = []
        for row in self.results:
            comments.append(row[1])
        return comments
    
class EntityRecognizer(RecognizerABC):
    """Recognizes entities using the standford nlp tools."""
    
    def __init__(self, stanford_nlp_server):
        self.stanford_nlp = stanford_nlp_server
            
    def __setup__(self):
        self.entities = Counter()
        
    def __extract_entities__(self, text):
        
        # Get the entities in a raw format
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            print(text)
        tagged_sentences = [self.stanford_nlp.ner(sentence) for sentence in sentences]
        
        # Return a list of the entities found
        entities = [pair[0]
            for tagged_sentence in tagged_sentences
            for pair in tagged_sentence if pair[1] in ('PERSON', 'LOCATION', 'ORGANIZATION')]
    
        return entities
        
    def __append_entities__(self, entities):
        self.entities.update(entities)
    
    def done(self):
        self.stanford_nlp.close()
    
class SentimentAnalyzer(AnalyzerABC):
    """Analyzes sentiments using the stanford nlp tools."""
    
    def __init__(self, core_nlp_server):
        self.core_nlp_server = core_nlp_server
        self.props = {'annotators': 'sentiment'}
        
        # Map from underlying engine tags to list indices
        self.class_mapper = {'verynegative': 0
                             ,'negative': 1
                             ,'neutral': 2
                             ,'positive': 3
                             ,'verypositive': 4}
    
    def __create_score__(self, result_dict):
        return_classes = [sentence['sentiment'].lower() for sentence in result_dict['sentences']]
        scores = [0]*len(self.class_mapper.keys())
        for return_class in return_classes:
            class_index = self.class_mapper[return_class]
            scores[class_index] += 1
        return scores
    
    def __analyze_text__(self, text):
        result = self.core_nlp_server.annotate(text, properties = self.props)
        result_dict = json.loads(result)
        return result_dict
    
    def __score_text__(self, text):
        result_dict = self.__analyze_text__(text)
        score = self.__create_score__(result_dict)
        return score
          
    def __mapper__(self):
        return self.class_mapper
        
class SentimentAnalysisController:
    """Chains entity recognition and semantic analysis together.
    
    Takes the given entity recognition system, sentiment analysis system, and
    a list of text snippets and aggregates sentiment scores for
    each entity found.
    """
    
    def __init__(self, recognition_engine, sentiment_engine, comments):
        
        try:
            self.recognizer = recognition_engine
            self.analyzer = sentiment_engine
            
            # Define how sentiment engine scores are translated from strings to numbers
            self.class_mapper = self.analyzer.mapper()
            
            # Define a standard bin to tally scores into
            default_list = [0]*len(self.class_mapper.keys())
            
            # Define defaul dict with the standard bin
            self.entity_sentiments = defaultdict(lambda: default_list)
            
            # Run the process
            self.__create_mapping__(comments)
        except:
            print_exc()
            
    def __add_sentiment_score__(self, entity, score):
        """Add the score to the entities dictionary entry."""
        
        entity_lowered = entity.lower()
        current_score = self.entity_sentiments[entity_lowered]
        new_score = list(map(add, current_score, score))
        self.entity_sentiments[entity_lowered] = new_score
    
    def __create_mapping__(self, comments):
        
        print('Starting analysis...')
        
        # Progress timer setup
        total_comments = len(comments)
        i = 1
        j = 0
        start_time = time.time()
        time_increment = 60
        
        for comment in comments:
            
            # Entity analyzer
            try:
                entities = self.recognizer.process_text(comment)
                if len(entities) > 0:
                    score = self.analyzer.score_text(comment)
                    for entity in entities:
                        self.__add_sentiment_score__(entity, score)
            except:
                print('Issue encountered when mapping comment starting with "' + comment[:20] + '..."')
            
            # Progress timer
            if time.time() - start_time >= time_increment*j:
                j+=1
                elapsed_time = round((time.time() - start_time)/60, 1)
                eta = round((total_comments - i)*(elapsed_time/i), 1)
                percentage = round(100*float(i)/total_comments, 1)
                print(str(percentage) + '% complete, elapsed time ' + str(elapsed_time) + 'm, ETA. ' + str(eta) + 'm')
            i+=1
            
            
        final_time = round((time.time() - start_time)/60, 1)
        print('Analysis complete... elapsed time ' + str(final_time) + 'm')
                    
    def scores(self):
        return self.entity_sentiments

def quick_run(subreddit, year, month, comment_amount, json_service_account, stanford_path):
    """A quick run of the sentiment analysis system on reddit
    
    subreddit - name of the subreddit you want to pull comments from
    year - year the comments were published
    comment_amount - amount of comments you wish to look at
    json_service_account - path to a json google service account file
    stanford_path - path to the StanfordCoreNLP library
    
    The system leverages NLTK, StanfordCoreNLP, and BigQuery to perform
    entity recognition and rudimentary sentiment analysis on those entities.
    """
	
    # Load the comment
    print('Loading comments...')
    client = GClient(json_service_account)
    loader = Loader(subreddit, year, month, comment_amount, client.client)
    
    # Setup underlying systems
    stanford_nlp = StanfordCoreNLP(stanford_path, memory = '8g')
    recognition_engine = EntityRecognizer(stanford_nlp)
    sentiment_engine = SentimentAnalyzer(stanford_nlp)
    
    # Perform analysis
    print('Starting analysis controller...')
    analysis = SentimentAnalysisController(recognition_engine, sentiment_engine
                                           ,loader.output_list())
    print('Analysis complete...')
	
    return analysis.scores()
