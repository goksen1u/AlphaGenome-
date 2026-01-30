# Guide Complet AlphaGenome - Comprendre Chaque Étape

## 📚 Table des Matières
1. [Introduction à AlphaGenome](#introduction)
2. [Contexte et Innovation](#contexte)
3. [Installation et Configuration](#installation)
4. [Comprendre le Notebook Quick Start](#notebook)
5. [Les Concepts Clés](#concepts)
6. [Utilisation Pratique](#pratique)
7. [Analyse des Variants](#variants)
8. [Ressources et Support](#ressources)

---

## 🧬 Introduction à AlphaGenome {#introduction}

### Qu'est-ce qu'AlphaGenome ?

AlphaGenome est un modèle d'intelligence artificielle développé par Google DeepMind qui prédit les fonctions et effets de séquences d'ADN. Publié dans Nature en janvier 2026, il représente une avancée majeure dans la compréhension du génome humain.

### Points clés :
- **Input** : Séquences ADN jusqu'à 1 million de paires de bases
- **Output** : Milliers de prédictions sur les propriétés fonctionnelles
- **Résolution** : Prédictions à la résolution d'une seule paire de bases
- **Performance** : Surpasse 25 des 26 modèles existants testés

### Pourquoi est-ce révolutionnaire ?

- **98% du génome** : AlphaGenome s'attaque aux régions non-codantes (98% de l'ADN) qui régulent l'expression des gènes
- **Modèle unifié** : Remplace plusieurs modèles spécialisés par un seul outil complet
- **Résolution sans précédent** : Combine longue séquence (1Mb) ET haute résolution (1bp)

---

## 🎯 Contexte et Innovation {#contexte}

### Le Problème Scientifique

Le génome humain contient 3,1 milliards de lettres (A, T, C, G), mais :
- Seulement 2% codent pour des protéines
- Les 98% restants régulent l'expression des gènes
- De petites variations peuvent causer des maladies

### L'Innovation AlphaGenome

**Avant** : Les modèles devaient choisir entre :
- Longues séquences MAIS basse résolution
- Haute résolution MAIS courtes séquences

**Avec AlphaGenome** :
- Séquences de 1 million de paires de bases
- Résolution à la paire de bases unique
- Entraînement en seulement 4 heures (vs Enformer qui nécessitait le double)

### Types de Prédictions

AlphaGenome prédit 11 types de modalités différentes :

1. **ATAC** - Accessibilité de la chromatine
2. **CAGE** - Initiation de la transcription
3. **DNASE** - Régions d'ADN accessibles
4. **RNA_SEQ** - Expression génique
5. **CHIP_HISTONE** - Modifications des histones
6. **CHIP_TF** - Liaison de facteurs de transcription
7. **SPLICE_SITES** - Sites d'épissage
8. **SPLICE_SITE_USAGE** - Utilisation des sites d'épissage
9. **SPLICE_JUNCTIONS** - Jonctions d'épissage
10. **CONTACT_MAPS** - Cartes de contacts 3D de la chromatine
11. **PROCAP** - Initiation de la transcription précise

---

## 💻 Installation et Configuration {#installation}

### Méthode 1 : Utilisation de l'API (Recommandée)

```bash
# Installer le package AlphaGenome
pip install alphagenome
```

**Avantages** :
- Pas besoin de GPU
- Accès immédiat au modèle
- Gratuit pour usage non-commercial
- ~1 million de requêtes/jour gérées

**Obtenir une clé API** :
1. Visiter https://github.com/google-deepmind/alphagenome
2. Suivre les instructions pour obtenir une clé API
3. Stocker la clé de manière sécurisée

### Méthode 2 : Installation Locale (Recherche Avancée)

```bash
# Cloner le dépôt de recherche
git clone https://github.com/google-deepmind/alphagenome_research.git
pip install -e ./alphagenome_research
```

**Requis** :
- GPU NVIDIA H100 (recommandé)
- CUDA et cuDNN installés
- JAX correctement configuré
- Télécharger les poids du modèle depuis Kaggle ou Hugging Face

---

## 📓 Comprendre le Notebook Quick Start {#notebook}

Analysons le notebook étape par étape :

### Étape 1 : Installation

```python
# Installation d'AlphaGenome
pip install alphagenome
```

**Ce qui se passe** : Installation du package Python qui contient le client API et les utilitaires.

---

### Étape 2 : Imports

```python
from alphagenome import colab_utils
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
import matplotlib.pyplot as plt
import pandas as pd
```

**Modules importés** :
- `colab_utils` : Utilitaires pour Google Colab (gestion clés API)
- `gene_annotation` : Annotations géniques (GENCODE, etc.)
- `genome` : Manipulation de séquences génomiques
- `transcript_utils` : Utilitaires pour les transcrits
- `ism` : In Silico Mutagenesis (mutations virtuelles)
- `dna_client` : Client principal pour le modèle
- `variant_scorers` : Scorage d'effets de variants
- `plot_components` : Visualisation des résultats

---

### Étape 3 : Charger le Modèle

```python
dna_model = dna_client.create(colab_utils.get_api_key())
```

**Ce qui se passe** :
1. Récupération de la clé API (depuis Colab Secrets ou variable)
2. Création d'une instance du client qui communique avec l'API AlphaGenome
3. Le modèle est maintenant prêt à faire des prédictions

---

### Étape 4 : Explorer les Types de Sortie

```python
[output.name for output in dna_client.OutputType]
```

**Résultat** :
```
['ATAC', 'CAGE', 'DNASE', 'RNA_SEQ', 'CHIP_HISTONE', 
 'CHIP_TF', 'SPLICE_SITES', 'SPLICE_SITE_USAGE', 
 'SPLICE_JUNCTIONS', 'CONTACT_MAPS', 'PROCAP']
```

**Signification** : Ce sont toutes les modalités que le modèle peut prédire.

---

### Étape 5 : Faire une Prédiction Simple

```python
output = dna_model.predict_sequence(
    sequence='GATTACA'.center(dna_client.SEQUENCE_LENGTH_1MB, 'N'),
    requested_outputs=[dna_client.OutputType.DNASE],
    ontology_terms=['UBERON:0002048'],  # Poumon
)
```

**Décortiquons cette commande** :

1. **`sequence='GATTACA'.center(dna_client.SEQUENCE_LENGTH_1MB, 'N')`**
   - Prend la séquence 'GATTACA'
   - La centre dans une séquence de 1Mb
   - Remplit avec des 'N' (nucléotides indéfinis)
   - `SEQUENCE_LENGTH_1MB` = 1,048,576 bases

2. **`requested_outputs=[dna_client.OutputType.DNASE]`**
   - Demande uniquement les prédictions DNase-seq
   - On peut en demander plusieurs : `[OutputType.DNASE, OutputType.RNA_SEQ]`

3. **`ontology_terms=['UBERON:0002048']`**
   - Filtre les prédictions pour le tissu pulmonaire
   - UBERON est une ontologie standardisée pour l'anatomie
   - Sans ce filtre, toutes les pistes tissu/cellule seraient prédites

**Résultat** : Un objet contenant les prédictions pour DNase dans le poumon.

---

### Étape 6 : Examiner l'Objet TrackData

```python
dnase = output.dnase
type(dnase)  # alphagenome.data.track_data.TrackData
```

**Structure d'un objet TrackData** :

```
TrackData
├── values: array numpy des prédictions (forme: [n_tracks, sequence_length])
├── tracks: DataFrame pandas avec métadonnées des pistes
│   ├── track_id
│   ├── tissue/cell_type
│   ├── experiment_type
│   └── ontology_terms
├── start_position: position de début dans le génome
└── end_position: position de fin dans le génome
```

**Propriétés importantes** :
- `dnase.values` : Valeurs numériques des prédictions
- `dnase.tracks` : Informations sur chaque piste
- `dnase.values.shape` : Dimensions (nombre de pistes × longueur de séquence)

---

### Étape 7 : Visualiser les Prédictions

```python
plot_components.plot_tracks(
    dnase,
    start=500_000,
    end=501_000,
    smooth_window=10
)
plt.show()
```

**Paramètres** :
- `start/end` : Région génomique à afficher (en paires de bases)
- `smooth_window` : Lissage des données (moyenne mobile)

**Le graphique montre** :
- Axe X : Position dans la séquence
- Axe Y : Signal prédit (intensité DNase)
- Chaque ligne : Une piste (tissu/cellule différent)

---

## 🔬 Les Concepts Clés {#concepts}

### 1. Termes d'Ontologie

AlphaGenome utilise des ontologies standardisées pour identifier les tissus/cellules :

**UBERON** (Anatomie) :
- `UBERON:0002048` → Poumon
- `UBERON:0000955` → Cerveau
- `UBERON:0002107` → Foie

**CL** (Types cellulaires) :
- `CL:0000236` → Lymphocyte B
- `CL:0000084` → Lymphocyte T

**Comment trouver les termes** :
```python
# Lister tous les termes disponibles pour un type de sortie
terms = dna_model.get_available_ontology_terms(
    output_type=dna_client.OutputType.DNASE
)
print(terms[:10])  # Afficher les 10 premiers
```

---

### 2. Longueurs de Séquence Valides

AlphaGenome accepte 3 longueurs de séquence :

```python
# Constantes disponibles
dna_client.SEQUENCE_LENGTH_256KB  # 262,144 bp
dna_client.SEQUENCE_LENGTH_512KB  # 524,288 bp
dna_client.SEQUENCE_LENGTH_1MB    # 1,048,576 bp
```

**Pourquoi ces longueurs spécifiques ?**
- Puissances de 2 pour efficacité computationnelle
- 1Mb peut capturer des régulations à longue distance
- Plus la séquence est longue, plus le contexte est riche

**Padding** :
```python
# Centrer une courte séquence
short_seq = "ATCGATCG"
padded = short_seq.center(dna_client.SEQUENCE_LENGTH_1MB, 'N')

# Ou tronquer une longue séquence
long_seq = genome_sequence[start:start+dna_client.SEQUENCE_LENGTH_1MB]
```

---

### 3. Résolution des Prédictions

La résolution varie selon la modalité :

| Modalité | Résolution | Exemple d'utilisation |
|----------|-----------|----------------------|
| DNASE, ATAC, CAGE | 128 bp | Identifier régions régulatrices larges |
| RNA_SEQ | 32 bp | Quantifier expression génique |
| CHIP_* | Variable | Localiser liaison protéines |
| SPLICE_SITES | 1 bp | Identifier sites exacts d'épissage |
| CONTACT_MAPS | Bins de 2kb | Comprendre structure 3D chromatine |

**Accéder à la résolution** :
```python
# La résolution est dans les métadonnées de la piste
resolution = dnase.tracks['resolution'].iloc[0]
print(f"Résolution: {resolution} bp")
```

---

## 🧪 Utilisation Pratique {#pratique}

### Cas d'Usage 1 : Analyser une Région Génomique Spécifique

```python
# 1. Charger une séquence génomique depuis un fichier FASTA
from alphagenome.data import genome

# Télécharger le génome de référence (hg38)
genome_data = genome.load_genome('hg38')

# Extraire une région d'intérêt (chromosome, début, fin)
sequence = genome_data.extract_sequence(
    chromosome='chr1',
    start=1_000_000,
    end=2_000_000
)

# 2. Faire des prédictions multi-modalités
output = dna_model.predict_sequence(
    sequence=sequence,
    requested_outputs=[
        dna_client.OutputType.RNA_SEQ,
        dna_client.OutputType.DNASE,
        dna_client.OutputType.CHIP_HISTONE
    ],
    ontology_terms=['UBERON:0002048']  # Poumon
)

# 3. Visualiser les résultats
fig, axes = plt.subplots(3, 1, figsize=(15, 10))

plot_components.plot_tracks(output.rna_seq, ax=axes[0])
axes[0].set_title('Expression ARN')

plot_components.plot_tracks(output.dnase, ax=axes[1])
axes[1].set_title('Accessibilité DNase')

plot_components.plot_tracks(output.chip_histone, ax=axes[2])
axes[2].set_title('Modifications Histones')

plt.tight_layout()
plt.savefig('region_analysis.png', dpi=300)
```

---

### Cas d'Usage 2 : Analyser un Gène Complet

```python
from alphagenome.data import gene_annotation

# 1. Charger les annotations géniques
annotations = gene_annotation.load_gencode('gencode.v44.annotation.gtf')

# 2. Trouver un gène d'intérêt (par exemple BRCA1)
gene = annotations.get_gene('BRCA1')

# 3. Obtenir la séquence du gène avec contexte régulatoire
# Ajouter 100kb upstream et downstream
sequence = genome_data.extract_sequence(
    chromosome=gene.chromosome,
    start=gene.start - 100_000,
    end=gene.end + 100_000
)

# 4. Prédire l'épissage et l'expression
output = dna_model.predict_sequence(
    sequence=sequence,
    requested_outputs=[
        dna_client.OutputType.SPLICE_SITES,
        dna_client.OutputType.SPLICE_JUNCTIONS,
        dna_client.OutputType.RNA_SEQ
    ]
)

# 5. Visualiser les isoformes prédits
from alphagenome.visualization import plot_transcript

plot_transcript.plot_gene_structure(
    gene=gene,
    splice_predictions=output.splice_sites,
    rna_predictions=output.rna_seq
)
```

---

### Cas d'Usage 3 : Comparer Plusieurs Tissus

```python
# 1. Définir les tissus à comparer
tissues = {
    'Poumon': 'UBERON:0002048',
    'Foie': 'UBERON:0002107',
    'Cerveau': 'UBERON:0000955',
    'Coeur': 'UBERON:0000948'
}

# 2. Faire des prédictions pour chaque tissu
results = {}
for tissue_name, ontology_term in tissues.items():
    output = dna_model.predict_sequence(
        sequence=sequence,
        requested_outputs=[dna_client.OutputType.RNA_SEQ],
        ontology_terms=[ontology_term]
    )
    results[tissue_name] = output.rna_seq

# 3. Créer une heatmap comparative
import seaborn as sns

# Extraire les valeurs moyennes par tissu
expression_data = {}
for tissue, track_data in results.items():
    # Moyenne sur toutes les pistes de ce tissu
    expression_data[tissue] = track_data.values.mean(axis=0)

# Créer DataFrame
df = pd.DataFrame(expression_data)

# Visualiser
plt.figure(figsize=(15, 8))
sns.heatmap(df.T, cmap='YlOrRd', robust=True)
plt.title('Expression Comparative entre Tissus')
plt.xlabel('Position Génomique')
plt.ylabel('Tissu')
plt.tight_layout()
plt.savefig('tissue_comparison.png', dpi=300)
```

---

## 🧬 Analyse des Variants {#variants}

### Qu'est-ce qu'un Variant ?

Un variant est une modification d'une ou plusieurs paires de bases dans l'ADN :
- **SNV** (Single Nucleotide Variant) : C → T
- **Insertion** : ATCG → ATCGAA
- **Délétion** : ATCGATCG → ATCG
- **Substitution** : ATCG → TTCG

### Scorage de Variants avec AlphaGenome

```python
from alphagenome.models import variant_scorers

# 1. Définir le variant
# Format : chromosome:position:ref>alt
variant = "chr17:43044295:G>A"  # Exemple dans BRCA1

# 2. Créer un variant scorer
scorer = variant_scorers.VariantScorer(
    dna_model=dna_model,
    scoring_method='ism'  # In Silico Mutagenesis
)

# 3. Scorer le variant
score = scorer.score_variant(
    variant=variant,
    output_types=[
        dna_client.OutputType.RNA_SEQ,
        dna_client.OutputType.SPLICE_SITES
    ],
    ontology_terms=['UBERON:0000955']  # Cerveau
)

# 4. Interpréter le score
print(f"Variant: {variant}")
print(f"Impact sur RNA-Seq: {score['RNA_SEQ']}")
print(f"Impact sur Épissage: {score['SPLICE_SITES']}")

# Score positif = augmentation de l'activité
# Score négatif = diminution de l'activité
# Score proche de 0 = peu d'impact
```

---

### Méthodes de Scorage

AlphaGenome propose plusieurs méthodes :

#### 1. ISM (In Silico Mutagenesis)

```python
# Compare la séquence de référence avec la séquence mutée
score = scorer.score_variant(
    variant=variant,
    scoring_method='ism'
)
```

**Comment ça marche** :
1. Prédiction sur séquence de référence → Pred_ref
2. Prédiction sur séquence mutée → Pred_mut
3. Score = Pred_mut - Pred_ref

**Avantages** :
- Simple et intuitif
- Rapide (2 prédictions seulement)

---

#### 2. Gradient-based Scoring

```python
# Utilise les gradients du modèle
score = scorer.score_variant(
    variant=variant,
    scoring_method='gradient'
)
```

**Comment ça marche** :
- Calcule l'importance de chaque position via les gradients
- Plus précis pour les effets subtils

---

#### 3. Saturation Mutagenesis

```python
from alphagenome.interpretation import ism

# Tester TOUTES les mutations possibles dans une région
region_sequence = sequence[500_000:501_000]  # 1kb région

saturation_results = ism.saturation_mutagenesis(
    dna_model=dna_model,
    sequence=region_sequence,
    output_type=dna_client.OutputType.DNASE,
    position_range=(0, len(region_sequence))
)

# Visualiser la carte de mutagénèse
plot_components.plot_saturation_mutagenesis(saturation_results)
```

**Résultat** :
- Une matrice 4 × longueur (A, T, C, G × positions)
- Montre l'effet de chaque mutation possible
- Identifie les positions critiques

---

### Analyse de Variants Multiples

```python
# Liste de variants à analyser (format VCF)
variants = [
    "chr17:43044295:G>A",
    "chr17:43044295:G>T",
    "chr17:43045802:C>T",
    # ... plus de variants
]

# Scorer tous les variants
scores_df = pd.DataFrame()

for variant in variants:
    score = scorer.score_variant(
        variant=variant,
        output_types=[dna_client.OutputType.RNA_SEQ]
    )
    
    scores_df = pd.concat([
        scores_df,
        pd.DataFrame({
            'variant': [variant],
            'rna_seq_score': [score['RNA_SEQ']],
            'impact_category': ['high' if abs(score['RNA_SEQ']) > 0.5 else 'low']
        })
    ])

# Sauvegarder les résultats
scores_df.to_csv('variant_scores.csv', index=False)

# Visualiser
plt.figure(figsize=(12, 6))
plt.barh(scores_df['variant'], scores_df['rna_seq_score'])
plt.axvline(x=0, color='black', linestyle='--')
plt.xlabel('Impact Score')
plt.title('Impacts des Variants sur l\'Expression ARN')
plt.tight_layout()
plt.savefig('variant_impacts.png', dpi=300)
```

---

### Exemple Clinique : Variant Pathogène

```python
# Analyser un variant connu pour causer une maladie
# Exemple : Variant dans le promoteur du gène HBB (bêta-globine)
# Associé à la thalassémie

# 1. Charger la région
hbb_sequence = genome_data.extract_sequence(
    chromosome='chr11',
    start=5_246_000,  # Région promotrice HBB
    end=5_248_000
)

# 2. Définir le variant pathogène
pathogenic_variant = "chr11:5246877:A>G"  # Exemple

# 3. Analyser l'impact multi-modal
score = scorer.score_variant(
    variant=pathogenic_variant,
    output_types=[
        dna_client.OutputType.RNA_SEQ,
        dna_client.OutputType.CAGE,  # Initiation transcription
        dna_client.OutputType.CHIP_TF,  # Liaison facteurs de transcription
        dna_client.OutputType.DNASE
    ],
    ontology_terms=['CL:0000232']  # Érythrocytes
)

# 4. Rapport détaillé
print("=" * 50)
print(f"Analyse du variant: {pathogenic_variant}")
print("=" * 50)
print(f"Impact sur expression ARN: {score['RNA_SEQ']:.3f}")
print(f"Impact sur initiation (CAGE): {score['CAGE']:.3f}")
print(f"Impact sur liaison TF: {score['CHIP_TF']:.3f}")
print(f"Impact sur accessibilité: {score['DNASE']:.3f}")
print("=" * 50)

# 5. Visualiser le contexte génomique
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Prédictions référence vs mutant
ref_output = dna_model.predict_sequence(
    sequence=hbb_sequence,
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
    ontology_terms=['CL:0000232']
)

# Créer séquence mutante
mut_sequence = apply_variant(hbb_sequence, pathogenic_variant)
mut_output = dna_model.predict_sequence(
    sequence=mut_sequence,
    requested_outputs=[dna_client.OutputType.RNA_SEQ],
    ontology_terms=['CL:0000232']
)

# Comparer
plot_components.plot_comparison(
    ref_output.rna_seq,
    mut_output.rna_seq,
    variant_position=variant_position,
    ax=axes[0]
)

plt.savefig('clinical_variant_analysis.png', dpi=300)
```

---

## 📚 Ressources et Support {#ressources}

### Documentation Officielle

1. **Site Principal** : https://www.alphagenomedocs.com/
2. **GitHub API** : https://github.com/google-deepmind/alphagenome
3. **GitHub Research** : https://github.com/google-deepmind/alphagenome_research
4. **Paper Nature** : https://www.nature.com/articles/s41586-025-10014-0
5. **Preprint bioRxiv** : https://doi.org/10.1101/2025.06.25.661532

### Tutoriels Interactifs

**Google Colab Notebooks** :
- Quick Start : Introduction de base
- Visualization : Apprendre à visualiser
- Advanced Scoring : Techniques avancées de scorage
- Genome Browser Integration : Intégration avec navigateurs génomiques

**Lien Colab** : Disponible sur le GitHub officiel

### Forums et Support

1. **Community Forum** : Forum officiel AlphaGenome
   - Questions d'utilisation
   - Partage d'expériences
   - Demandes de fonctionnalités

2. **GitHub Issues** : Pour bugs et problèmes techniques
   - https://github.com/google-deepmind/alphagenome/issues

3. **Email Support** : alphagenome@google.com
   - Pour questions complexes
   - Collaborations

### Datasets et Ressources

**Données d'Entraînement** :
- ENCODE : https://www.encodeproject.org/
- GTEx : https://gtexportal.org/
- 4D Nucleome : https://www.4dnucleome.org/
- FANTOM5 : https://fantom.gsc.riken.jp/

**Annotations Génomiques** :
- GENCODE : https://www.gencodegenes.org/
- RefSeq : https://www.ncbi.nlm.nih.gov/refseq/
- Ensembl : https://www.ensembl.org/

**Génomes de Référence** :
- hg38 (humain) : https://hgdownload.soe.ucsc.edu/
- mm10 (souris) : https://hgdownload.soe.ucsc.edu/

### Exemples de Projets

**1. Identifier variants régulateurs dans cancer**
```python
# Analyser mutations somatiques dans promoteurs/enhancers
# Prioriser variants driver vs passenger
```

**2. Prédire effets épissage pour maladies rares**
```python
# Scorer variants dans sites d'épissage
# Identifier variants cryptiques
```

**3. Caractériser variants GWAS non-codants**
```python
# Interpréter SNPs associés à maladies complexes
# Identifier tissus/cellules affectés
```

### Limitations à Connaître

1. **Espèces** : Entraîné uniquement sur humain et souris
2. **Faux négatifs** : Peut manquer certains effets subtils
3. **Contexte cellulaire** : Limité aux types de cellules dans les données d'entraînement
4. **Non clinique** : Outil de recherche, pas pour diagnostic médical
5. **Variants complexes** : Performance variable sur insertions/délétions longues

### Meilleures Pratiques

1. **Validation Expérimentale** :
   - Toujours valider les prédictions importantes en labo
   - AlphaGenome est un outil de priorisation, pas de vérité absolue

2. **Interprétation Prudente** :
   - Considérer le contexte biologique
   - Croiser avec autres sources de données

3. **Utilisation Efficace de l'API** :
   - Batchez les requêtes quand possible
   - Utilisez les filtres ontology_terms pour réduire le compute

4. **Documentation** :
   - Documentez vos analyses
   - Citez AlphaGenome correctement dans publications

### Citation

```bibtex
@article{alphagenome2026,
  title={Advancing regulatory variant effect prediction with AlphaGenome},
  author={Avsec, Žiga and Latysheva, Natasha and Cheng, Jun and others},
  journal={Nature},
  volume={649},
  number={8099},
  year={2026},
  doi={10.1038/s41586-025-10014-0},
  publisher={Nature Publishing Group UK London}
}
```

---

## 🎓 Exercices Pratiques

### Exercice 1 : Première Prédiction

**Objectif** : Faire votre première prédiction AlphaGenome

```python
# TODO: 
# 1. Installer alphagenome
# 2. Obtenir une clé API
# 3. Prédire DNase sur une petite séquence
# 4. Visualiser le résultat
```

### Exercice 2 : Analyser un Gène

**Objectif** : Analyser l'expression d'un gène dans différents tissus

```python
# TODO:
# 1. Choisir un gène (ex: TP53, BRCA1, MYC)
# 2. Extraire sa séquence
# 3. Prédire RNA-seq dans 5 tissus différents
# 4. Comparer les résultats
```

### Exercice 3 : Scorer un Variant

**Objectif** : Évaluer l'impact d'un variant génétique

```python
# TODO:
# 1. Trouver un variant dans dbSNP
# 2. Le scorer avec AlphaGenome
# 3. Interpréter le score
# 4. Comparer avec annotations cliniques existantes
```

---

## 🔍 Dépannage

### Problème : "API Key Invalid"

**Solution** :
```python
# Vérifier que la clé est correctement stockée
import os
api_key = os.environ.get('ALPHAGENOME_API_KEY')
print(f"Clé trouvée: {api_key is not None}")
```

### Problème : "Sequence length invalid"

**Solution** :
```python
# Utiliser les constantes prédéfinies
from alphagenome.models import dna_client

valid_lengths = [
    dna_client.SEQUENCE_LENGTH_256KB,
    dna_client.SEQUENCE_LENGTH_512KB,
    dna_client.SEQUENCE_LENGTH_1MB
]
print(f"Longueurs valides: {valid_lengths}")
```

### Problème : "Rate limit exceeded"

**Solution** :
- Attendre quelques minutes
- Batchez vos requêtes
- Considérez l'installation locale pour analyses à grande échelle

### Problème : "Out of memory" (installation locale)

**Solution** :
```python
# Utiliser des séquences plus courtes
# Ou augmenter la RAM GPU disponible
# Ou utiliser l'API au lieu de l'installation locale
```

---

## 📈 Prochaines Étapes

1. **Commencer simple** : Essayez le notebook Quick Start
2. **Explorer** : Testez différentes modalités et tissus
3. **Approfondir** : Analysez vos propres régions génomiques
4. **Scorer variants** : Évaluez des variants d'intérêt
5. **Contribuer** : Partagez vos découvertes avec la communauté

---

## 💡 Conseils Finaux

- **Soyez patient** : La biologie est complexe, prenez le temps de comprendre
- **Expérimentez** : N'hésitez pas à tester différentes approches
- **Documentez** : Gardez trace de vos analyses
- **Partagez** : La science avance par la collaboration
- **Restez curieux** : AlphaGenome est un outil puissant pour explorer le génome

---

**Bonne exploration avec AlphaGenome ! 🧬🔬**

*Guide créé le 29 janvier 2026*
*Basé sur la publication Nature et la documentation officielle*
