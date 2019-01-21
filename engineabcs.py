from abc import ABC, abstractmethod

class RecognizerABC(ABC):
    """Interface definition for entity recognizer modules."""
    
    @abstractmethod
    def __extract_entities__(self, text):
        """Returns entities from text
        
        This method should be able to handle sentences, paragraphs, or larger
        pieces of text and return the entities contained within it. It must
        return a list of named entities.
        """
        
        pass
        
    def process_text(self, text):
        """Returns the entities created by recognizer
        
        This is the main method other systems use to interact with the
        underlying entity recognition system of choice.
        """
        
        entities = self.__extract_entities__(text)
        assert isinstance(entities, list)
        
        return entities

class AnalyzerABC(ABC):
    """Interface definition for sentiment analysis modules."""
    
    @abstractmethod
    def __score_text__(self, text):
        """Return the score of some input text"""
        
        pass
    
    @abstractmethod
    def __mapper__(self):
        """Returns a mapping dictionary
        
        The mapping dictionary converts the output of the underlying sentiment
        analysis system to sequential integer valued scores.
        """
        
        pass
    
    def score_text(self, text):
        
        sentiment_scores = self.__score_text__(text)
        assert isinstance(sentiment_scores, list)
        assert len(sentiment_scores) == len(self.mapper())
        return sentiment_scores
          
    def mapper(self):
        
        assert isinstance(self.__mapper__(), dict)
        return self.__mapper__()