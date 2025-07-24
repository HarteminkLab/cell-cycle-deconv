

from goatools.obo_parser import GODag
from collections import defaultdict
from goatools.godag_plot import plot_gos, plot_results, plot_goid2goobj
from goatools.go_enrichment import GOEnrichmentStudy
import pandas
from src.sgd import read_sgd_w_go
from src.geneset import get_deconvolved_geneset
from src.utils import print_fl


class GeneOntology:

	def __init__(self, go_obo_path='./data/goslim_yeast.obo'):

		self.obodag = GODag(go_obo_path)

		# read genes containing GO Ontology annotations
		orfs_with_go = read_sgd_w_go()

		# only use canonical orfs dataset
		paper_geneset = get_deconvolved_geneset()
		self.orfs_with_go = orfs_with_go.loc[paper_geneset.index]

		# create mapping of gene names to set of GO annotaitons
		assoc = defaultdict(set)
		for idx, gene in self.orfs_with_go.iterrows():
			assoc[gene['name']] = set(gene.ontology.split(','))
		self.assoc = assoc
		self.methods = ['fdr_bh', 'bonferroni']

		self.devnull = open('/dev/null', 'w')

		# create GO enrichment object to run GO
		self.goeaobj = GOEnrichmentStudy(
			assoc.keys(), # List of protein-coding genes
			assoc, # geneid/GO associations
			self.obodag, # Ontologies
			propagate_counts=False,
			alpha=0.05, # default significance cut-off
			methods=self.methods,
			log=self.devnull)

	def run_go(self, geneids, sig=0.001):
		"""Run gene ontology against set of genes"""
		self.goea_results_all = self.goeaobj.run_study(geneids)
		self.goea_results_sig = [r for r in self.goea_results_all if (r.p_fdr_bh < sig and r.study_count > 0)]

		cols = ['id', 'name', 'pop_count', 'pop_n', 
				'study_count','study_n', 'pop_items', 'study_items'] + self.methods

		results_dic = {}

		for c in cols:
			results_dic[c] = []

		for g in self.goea_results_all:

			study_items = ','.join(g.study_items)
			name = g.name
			fdr = g.p_fdr_bh
			pop_items = ','.join(g.pop_items)

			results_dic['id'].append(g.GO)
			results_dic['name'].append(name)
			
			for method in self.methods:
				results_dic[method].append(g.__dict__['p_' + 
					method.replace('-', '_')])

			results_dic['study_items'].append(study_items)
			results_dic['pop_items'].append(pop_items)
			results_dic['study_count'].append(g.study_count)
			results_dic['pop_count'].append(g.pop_count)
			results_dic['study_n'].append(g.study_n)
			results_dic['pop_n'].append(g.pop_n)

		results_df = pandas.DataFrame(results_dic)
		self.results_df = results_df[cols].sort_values('fdr_bh').reset_index(drop=True)
		self.results_sig_df = self.results_df[
			(self.results_df.fdr_bh < sig) & 
			(self.results_df.study_count > 0)
			]

	def plot_sig(self):
		plot_results("test_{NS}.pdf", self.goea_results_sig)


def print_terms(obodag, terms):
	"""Print gene ontology terms from ontology ids and graph"""
	for gpar in terms:
		p = obodag.query_term(gpar)
		print_fl(gpar + " - " + p.name.title())


def get_term_names(obodag, terms):
	"""Print gene ontology terms from ontology ids and graph"""
	names = []
	for gpar in terms:
		p = obodag.query_term(gpar)
		names.append(p.name.title())

	return names


def genes_for_go(go_genes, go_terms):

	if type(go_terms) == str:
		go_mask = go_genes.ontology.str.contains(go_terms)
	else:
		go_mask = False
		for go_term in go_terms:
			go_mask = go_mask | go_genes.ontology.str.contains(go_term)

	return go_genes[go_mask]

