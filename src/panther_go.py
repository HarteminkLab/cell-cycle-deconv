
import pandas as pd
import requests


class PantherDBGO(object):
	"""docstring for PantherDBGO"""
	def __init__(self):
		
		reference_organism_id = 559292
		self.base_url = (f"https://pantherdb.org/services/oai/pantherdb/enrich/overrep"
			  f"?organism={reference_organism_id}&refOrganism={reference_organism_id}&annotDataSet=GO%3A0008150&"
			  f"enrichmentTestType=FISHER&correction=FDR&geneInputList=")

	def run_go(self, selected_gene_names):

		gene_list_str = ','.join(selected_gene_names.values.astype(str))
		self.gene_list = gene_list_str

		self.request_url = self.base_url + gene_list_str
		self.response_json = requests.get(self.request_url).json()

	def parse_response(self):

		results_df = pd.DataFrame(self.response_json['results']['result'])

		go_ids = results_df['term'].apply(lambda dic: dic['id'] if 'id' in dic.keys() else None)
		go_labels = results_df['term'].apply(lambda dic: dic['label'] if 'label' in dic.keys() else None)

		results_df['id'] = go_ids
		results_df['label'] = go_labels

		results_df = results_df[['id', 'label', 'number_in_list', 'number_in_reference', 'expected', 'fold_enrichment', 'fdr', 'pValue']]
		results_df = results_df[results_df.fdr < 0.05]

		self.results_df = results_df
