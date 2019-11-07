import os,sys,re
import numpy as np
import spacy as sc
import pandas as pd 
import nltk
import textacy
from textacy.extract import subject_verb_object_triples
import json
import collections
import pickle
import dill
import neuralcoref
import yake
from matplotlib import pyplot as plt


class TweetSVO():
	def __init__(self):
		self.nlp = sc.load("en_core_web_md")
		neuralcoref.add_to_pipe(self.nlp)
		self.spacy_stopwords = sc.lang.en.stop_words.STOP_WORDS
		self.pat_punct_remove=re.compile(r"[^\w' ]+")
		self.pat_extra_space=re.compile(r" {2,}")
		self.pat_url=re.compile(r"http[sS]\:\/\/[a-zA-Z0-9\.\/]+")
		self.pat_punct=re.compile(r"\-\_\"")
		self.pat_brackets=re.compile(r"\([a-zA-Z0-9\.]+\)")
		self.pat_convert_nonascii=re.compile(r'[^\x00-\x7f]')

	def clean_text(self,x):
		clean_x=re.sub(self.pat_convert_nonascii,r" ",x)
		clean_x=re.sub(self.pat_brackets,r"",clean_x)
		clean_x=re.sub(self.pat_punct,r" ",clean_x).lower()
		clean_x=re.sub(r"\n",r" ",clean_x)
		return clean_x

	def get_keywords_weights(self,words_list,vocab):
		words_weights_dict=collections.defaultdict(float)
		#assign weights to words in the text= equal weights to all words in the phrase, weight=1-yake_weight
		for kk in words_list:#mix of keywords and keyphrases.
			for ww in vocab:
				if ww in kk[0] and len(ww)>1:
					words_weights_dict[ww]+=(1.0-kk[1])
		return words_weights_dict

	def get_top_svo_sentences(self,svos,words_weights,text_st,k):
		svo_weights_mapping=collections.defaultdict(float)
		sentence_svo_map=collections.defaultdict(list)
		for ids,svo in enumerate(svos):
			tt=""
			svo_weight=0.0
			for sv in svo:
			    tt += sv.text +" "
			tt=tt.strip().split(" ")
			for ttw in tt:
			    svo_weight+=words_weights[ttw]
			svo_weight /= len(tt)
			svo_weights_mapping[svo]+= svo_weight
			sentence_svo_map[svo]=self.get_svo_sentences(svo,text_st)
		svo_weights_mapping=dict(sorted(svo_weights_mapping.items(),key=lambda x:x[1],reverse=True))
		svo_collection=collections.Counter(svo_weights_mapping) 
		top_k=svo_collection.most_common(k)
		# print(top_k)
		# print(sentence_svo_map)
		#find corresponding sentences and return  
		return [sentence_svo_map[tk[0]][0][1] for tk in top_k]

	def get_svo_sentences(self,svo,sc_text):
		sentence_svo_list=[]
		regg='%s.*?%s.*?%s' % (svo[0].text , svo[1].text,svo[2].text )
		pat_svo=re.compile(r'%s' % regg)
		for idn,ssn in enumerate(sc_text.sents):
			if len(pat_svo.findall(ssn.text))>0:
				sentence_svo_list.append((idn,ssn))
		return sentence_svo_list

if ( __name__ == "__main__"):
	# input: news article 
	# output: candidate sentences for tweet


	# flg_file=sys.argv[1]
	# if flg_file == "-f":
	#     input_file=sys.argv[2]
	#     with open(input_file,"r") as fin:
	#     	input_text=" ".join(fin.readlines()).strip()
	# else:
	# 	input_text=sys.argv[2]
	# k_num=sys.argv[3]


	svogetter=TweetSVO()


	# input_text="Among other recommendations in the report, the ACCC the ACCC said the ACCC wanted privacy law updated to give people the right to erase personal data stored online, aligning Australia with some elements of the European Union’s General Data Protection Regulation."
	input_text='SYDNEY (Reuters) - Australia said it would establish the world’s first dedicated office to police Facebook Inc (FB.O) and Google (GOOGL.O) as part of reforms designed to rein in the U.S. technology giants, potentially setting a precedent for global lawmakers.\n \n The move tightens the regulatory screws on the online platforms, which have governments from the United States to Europe scrambling to address concerns ranging from anti-trust issues to the spread of “fake news” and hate speech.\n \n Australian Treasurer Josh Frydenberg said the $5 billion fine slapped on Facebook in the United States this month for privacy breaches showed regulators were now taking such issues extremely seriously.\n \n “These companies are among the most powerful and valuable in the world,” Frydenberg told reporters in Sydney after the release of a much-anticipated report on future regulation of the dominant digital platforms.\n \n “They need to be held to account and their activities need to be more transparent.”\n \n Canberra would form a special branch of the Australian Competition and Consumer Commission (ACCC), the antitrust watchdog, to scrutinize how the companies used algorithms to match advertisements with viewers, giving them a stronghold on the main income generator of media operators.\n \n The new office was one of 23 recommendations in the ACCC’s report, including strengthening privacy laws, protections for the news media and a code of conduct requiring regulatory approval to govern how internet giants profit from users’ content.\n \n Frydenberg said the government intended to “lift the veil” on the closely guarded algorithms the firms use to collect and monetize users’ data, and accepted the ACCC’s “overriding conclusion that there is a need for reform”.\n \n The proposals would be subject to a 12-week public consultation process before the government acts on the report, he added.\n \n Google and Facebook have opposed tighter regulation while traditional media owners, including Rupert Murdoch’s News Corp (NWSA.O), have backed reform.\n \n News Corp’s local executive chairman, Michael Miller, welcomed the “strength of the language and the identification of the problems”, and said the publisher would work with the government to ensure “real change”.\n \n Facebook and Google said they would engage with the government during the consultation process, but had no comment on the specific recommendations.\n \n FILE PHOTO: The Google logo is pictured at the entrance to the Google offices in London, Britain January 18, 2019. REUTERS/Hannah McKay/File Photo\n \n The firms have previously rejected the need for tighter regulation and said the ACCC had underestimated the level of competition for online advertising.\n \n FIVE INVESTIGATIONS ONGOING\n \n ACCC Chairman Rod Sims said the regulator had five investigations of the two companies under way, and “I believe more will follow”.\n \n He said he was shocked at the amount of personal data the firms collected, often without users’ knowledge.\n \n “There needs to be a lot more transparency and oversight of Google and Facebook and their operations and practices,” he said.\n \n Among other recommendations in the report, the ACCC said it wanted privacy law updated to give people the right to erase personal data stored online, aligning Australia with some elements of the European Union’s General Data Protection Regulation.\n \n “We cannot leave these issues to be dealt with by commercial entities with substantial reach and market power. It’s really up to government and regulators to get up to date and stay up to date in relation to all these issues,” Sims said.\n \n FILE PHOTO: Silhouettes of mobile users are seen next to a screen projection of the Facebook logo in this picture illustration taken March 28, 2018. REUTERS/Dado Ruvic/Illustration/File Photo\n \n While the regulator did not recommend breaking up the tech giants, Sims also did not rule it out.\n \n “If it turns out that ... divestiture is a better approach, then that can always be recommended down the track,” he said.'
	k_num=3

	if isinstance(input_text, str):#if not spacy type then convert
		input_text=svogetter.nlp(input_text)

	#resolve coreferences
	text_coref_resolved=svogetter.nlp(input_text._.coref_resolved)
	
	#extract subject-verb-object tuples
	svo_list=list(textacy.extract.subject_verb_object_triples(text_coref_resolved))
	#get keywords and frequencies
	words=set([token.text for token in text_coref_resolved if len(token.text.strip())>0])
	kw=yake.KeywordExtractor(top=len(words))
	kws=kw.extract_keywords(text_coref_resolved.text)#try with coreference not resolved text input_text.text
	#get keywords frequencies
	words_weights=svogetter.get_keywords_weights(kws,words)

	#get sentences for top SVOs
	candidate_sentences=svogetter.get_top_svo_sentences(svo_list,words_weights,text_coref_resolved,k_num)
	print(candidate_sentences)