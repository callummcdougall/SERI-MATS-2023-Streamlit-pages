import torch as t
from transformer_lens import HookedTransformer
from tqdm import tqdm

# %pip install nltk
# %pip install --no-deps pattern

MY_SUFFIXES = ["r", "ic", "al", "ous", "able", "ful", "ive"]

def concat_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def verb_to_ing(verb):
    if len(verb) <= 2:
        return []
    else:
        return [
            verb[:-1] + 'ing',
            verb[:-2] + 'ying',
            verb + verb[-1] + 'ing',
            verb + 'ing',
        ]

def verb_to_noun(verb):
    if len(verb) <= 2:
        return []
    else:
        return [
            verb[:-1] + 'er',
            verb[:-1] + 'ion',
            verb[:-1] + 'ers',
            verb[:-1] + 'ions',
        ]

def noun_to_adj(noun):
    return [noun + suffix for suffix in MY_SUFFIXES] + [noun + suffix + "s" for suffix in MY_SUFFIXES]

def is_token(t1, model: HookedTransformer):
    '''For some reason this works, and "t1 is in model.tokenizer.vocab" doesn't.'''
    return len(model.to_str_tokens(t1, prepend_bos=False)) == 1
    # _t1 = t1 if not(t1.startswith(" ")) else "Ġ" + t1[1:]
    # return _t1 in model.tokenizer.vocab


def get_list_with_no_repetitions(L):
    '''Returns a list with no repetitions, preserving order. Also filters out anything of length < 2.'''
    L_no_repetitions = []
    for l in L:
        if l not in L_no_repetitions:
            L_no_repetitions.append(l)
    return L_no_repetitions



def get_equivalency_toks(t1: str, model: HookedTransformer, full_version: bool = False):
    
    # (C) add all syntactically related versions
    t1_orig = t1
    if full_version:
        # Full version where you use nltk's semantically related tokens
        t1_list = [t1.strip(" ")] + get_related_words(t1.strip(" ").lower(), model)
    else:
        # Mini version where you only look at plurals
        t1_stripped = t1.strip(" ").lower()
        t1_list = [
            t1_stripped,
            t1_stripped[:-1] if t1_stripped.endswith("s") else t1_stripped + "s",
        ]
    # (A) add capitalized versions
    t1_list = concat_lists([t1_list, [t1.capitalize() for t1 in t1_list], [t1.upper() for t1 in t1_list]])
    # (B) add versions with prepended spaces
    t1_list = t1_list + [" " + t1 for t1 in t1_list]
    # (D) replace all elements in t1_list with their tokenized versions (sorting so that non-split are first)
    # (this is also a bit hacky)
    t1_list_tokenized = model.to_str_tokens(t1_list, prepend_bos=False)

    # Get all tokens which are actual single tokens
    t1_list_single_toks = get_list_with_no_repetitions([i[0] for i in t1_list_tokenized if len(i) == 1])
    
    # Get all tokens which are tokenized versions of "core words"
    # What are "core words"? (1) the +space version, (2) the -space version (if it's a capital)
    # This captures things like " Berkeley"->"keley", but sadly not " Saskatchewan"->" Saskatchewans"
    # Also, I want to make sure that all the substrings are at least length 3, because if e.g. the capital letter gets stripped off then that probably means this isn't a useful substring (not really sure about this)
    core_words = []
    if not(t1_orig.startswith(" ")): core_words.append(t1_orig)
    if t1_orig.startswith(" ") and (len(t1_orig) > 1) and t1_orig[1].isupper(): core_words.append(t1_orig[1:])
    t1_list_multi_toks = model.to_str_tokens(core_words, prepend_bos=False)
    t1_list_multi_toks = get_list_with_no_repetitions([
        substr for tokenized_str in t1_list_multi_toks for substr in tokenized_str
        if len(tokenized_str) > 1 and min(map(len, tokenized_str)) >= 3
    ])
    t1_list_multi_toks = get_list_with_no_repetitions(concat_lists([i for i in t1_list_multi_toks if len(i) > 1]))

    return t1_list_single_toks, t1_list_multi_toks


def make_list_correct_length(L, K):
    '''
    If len(L) < K, pad list L with its last element until it is of length K.
    If len(L) > K, truncate.

    Special case when len(L) == 0, we just put the BOS token in it.
    '''
    if len(L) == 0:
        L = ["<|endoftext|>"]

    if len(L) <= K:
        L = L + [L[-1]] * (K - len(L))
    else:
        L = L[:K]

    assert len(L) == K
    return L



def create_full_semantic_similarity_dict(model: HookedTransformer, full_version: bool = False):
    '''
    Returns a full dictionary of semantic similarities.

    The keys are str_toks, the values are 3 lists:

        1. Equivalence-relation semantic similarity
            = things which are equivalent to each other (starting with the key itself)
        
        2. Superstrings
            = things which contain the key as a substring

        3. Substrings
            = things which are contained in the key as a substring

    When it comes to the actual CSPA function, we pick from (1) first, then (2), then (3).
    The reason we pick (2) over (3) is because, for example, we're more likely to find that
    "keley" is the source token and " Berkeley" is the token being predicted, than vice versa.
    The reverse is less interesting because it's probably bigram-y.

    Below this function, I've copied some examples.

    What are the biggest problems with this method?

        (1) Misses out some important semantic things, e.g.
                " write", " writing" and " writer"
                " rental", " rented" and " renting"
            
        (2) Misses some important non-semantic things, e.g. 1984 and 1985. However, hopefully 
            this doesn't matter because their similarity is mostly captured by the cosine 
            similarity of their unembeddings.
    '''
    # Load nltk's wordnet
    if full_version:
        import nltk
        from nltk.corpus import wordnet
        nltk.download('wordnet')
        
    # First, create a dictionary `d` which only contains lists (1) and (3)
    cspa_semantic_dict_reversed = {}
    all_vocab = model.to_str_tokens(t.arange(model.cfg.d_vocab).int())
    for str_tok in tqdm(all_vocab):
        # Weirdly specific TL bug, comes from characters like "の�"
        if model.to_tokens(str_tok, prepend_bos=False).numel() == 1:
            cspa_semantic_dict_reversed[str_tok] = get_equivalency_toks(str_tok, model, full_version)

    # Repair the dictionary with the weird tokens
    tokenizer_vocab_set = set([k if not(k.startswith("Ġ")) else " " + k[1:] for k in model.tokenizer.vocab.keys()])
    cspa_semantic_dict_reversed.update({k: ([k], []) for k in tokenizer_vocab_set - set(cspa_semantic_dict_reversed.keys())})

    # Construct the full dict from the reversed dict
    cspa_semantic_dict = {str_tok: [[str_tok], [], []] for str_tok in cspa_semantic_dict_reversed}
    for str_tok, (str_tok_single_list, str_tok_multi_list) in cspa_semantic_dict_reversed.items():
        for str_tok_single in str_tok_single_list:
            if str_tok_single in cspa_semantic_dict: cspa_semantic_dict[str_tok_single][0].append(str_tok)
        for str_tok_multi in str_tok_multi_list:
            if str_tok_multi in cspa_semantic_dict: cspa_semantic_dict[str_tok_multi][1].append(str_tok)
        cspa_semantic_dict[str_tok][2].extend(str_tok_multi_list)

    # Remove repetitions (and sort the substring list so that the longest are first)
    for k, v_list in cspa_semantic_dict.items():
        v_list[0] = get_list_with_no_repetitions(v_list[0])
        v_list[1] = get_list_with_no_repetitions([v for v in v_list[1] if v not in v_list[0]])
        v_list[2] = sorted(get_list_with_no_repetitions([v for v in v_list[2] if v not in v_list[0] + v_list[1]]), key=lambda x: len(x), reverse=True)
        cspa_semantic_dict[k] = v_list

    # Do the messy thing where the right things come first (eventually I'll ditch this and replace it with just having more projections)
    for k, v in cspa_semantic_dict.items():
        # Make sure the word itself is first
        v[0].remove(k)
        v[0].insert(0, k)
        # If the space-variant version is in v[0], make sure it's second
        space_variant = k[1:] if k.startswith(" ") else f" {k}"
        if space_variant in v[0]:
            v[0].remove(space_variant)
            v[0].insert(1, space_variant)
        # If the plural-variant is in v[0], and there are more than 2 items, make sure it's third
        plural_variant = k[:-1] if k.endswith("s") else f"{k}s"
        plural_withspace = plural_variant if plural_variant.startswith(" ") else f" {plural_variant}"
        plural_withoutspace = plural_variant if not(plural_variant.startswith(" ")) else plural_variant[1:]
        if plural_withspace in v[0]:
            v[0].remove(plural_withspace)
            v[0].insert(2, plural_withspace)
        if plural_withoutspace in v[0]:
            v[0].remove(plural_withoutspace)
            v[0].insert(3, plural_withoutspace)
    
    return cspa_semantic_dict_reversed, cspa_semantic_dict





#      ' Berkeley' -> [[' Berkeley'], [], ['ber', 'keley', 'ke', 'leys', 'Ber', 'BER', 'KE', 'LEY', 'LE', 'YS', ' ber', ' Ber', ' B', 'ER']]
#          'keley' -> [['keley'], [' Berkeley'], ['ke', 'leys', 'Ke', 'ley', 'KE', 'LEY', 'LE', 'YS', ' ke', ' Ke', ' KE']]
#    ' University' -> [[' University', ' university', 'University'], [], ['un', 'iversity', 's', 'UN', 'IVERS', 'ITY', 'S', ' UNIVERS']]
#          ' Mary' -> [[' Mary', 'mary', 'Mary'], [' Maryland'], ['s', 'M', 'ARY', 'S', ' m', 'ary', ' M']]
#          ' Pier' -> [[' Pier', ' pier'], [' Pierre', ' Pierce', ' piercing', 'Pierre', ' pierced', ' Piercing'], ['p', 'ier', 'iers', 'P', 'PI', 'ER', 'ERS', ' p', '  ...
#          ' pier' -> [[' pier', ' Pier'], [' Pierre', ' Pierce', ' piercing', 'Pierre', ' Piercing'], ['p', 'ier', 'iers', 'P', 'PI', 'ER', 'ERS', ' p', ' P', ' PI']]
#             'NY' -> [['NY', 'ny', ' NY', ' Ny'], ['anny', ' funny', 'enny', ' Danny', 'unny', ' NYC', 'nyder', ' Kenny', ' Snyder', ' Penny', ' penny', ' NYPD', ' sun ...
#          ' ring' -> [[' ring', 'ring', ' Ring', ' rings', ' Rings', 'rings', 'Ring'], ['rington', ' ringing', 'ringe'], ['R', 'ings', 'ING', 'INGS', ' R']]
#          ' Sask' -> [[' Sask'], [' Saskatchewan'], ['s', 'ask', 'asks', 'S', 'AS', 'K', 'KS', ' s', ' S', ' SAS']]
#  ' Saskatchewan' -> [[' Saskatchewan'], [], ['s', 'ask', 'atchewan', 'atche', 'w', 'ans', 'S', 'AS', 'K', 'ATCH', 'EW', 'AN', 'ANS', ' s', ' Sask', ' SAS']]
#             ' W' -> [[' W', 'W', 'w', ' w', 'ws', 'WS', ' WS', 'Ws'], [' with', ' we', ' wor', ' will', ' whe', ' were', ' would', 'wn', 'we', ' want', ' We', 'ward', ...


# And the first 3, which get used in the CSPA alg:

#      ' Berkeley' -> [' Berkeley', 'ber', 'keley']
#          'keley' -> ['keley', ' Berkeley', 'ke']
#    ' University' -> [' University', ' university', 'University']
#          ' Mary' -> [' Mary', 'mary', 'Mary']
#          ' Pier' -> [' Pier', ' pier', ' Pierre']
#          ' pier' -> [' pier', ' Pier', ' Pierre']
#             'NY' -> ['NY', 'ny', ' NY']
#          ' ring' -> [' ring', 'ring', ' Ring']
#          ' Sask' -> [' Sask', ' Saskatchewan', 's']
#  ' Saskatchewan' -> [' Saskatchewan', 's', 'ask']
#             ' W' -> [' W', 'W', 'w']





def get_related_words(word: str, model: HookedTransformer):
    from pattern.text.en import conjugate, PRESENT, PAST, FUTURE, SUBJUNCTIVE, INFINITIVE, PROGRESSIVE, PLURAL, SINGULAR
    from nltk.stem import WordNetLemmatizer
    MY_TENSES = [PRESENT, PAST, FUTURE, SUBJUNCTIVE, INFINITIVE, PROGRESSIVE]
    MY_NUMBERS = [PLURAL, SINGULAR]

    # Get stripped version (e.g. "writer" or "writing" -> "write")
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        ends_with_suffix = [suffix for suffix in MY_SUFFIXES if word.endswith(suffix)]
        if len(ends_with_suffix) > 0:
            lemma = word[:-len(ends_with_suffix[0])]
    related_words = []

    # Get things which pattern.en is pretty good with
    for tense in MY_TENSES:
        related_words.append(conjugate(lemma, tense=tense))
    for number in MY_NUMBERS:
        related_words.append(conjugate(lemma, number=number))

    # Get hardcoded things which pattern.en can't handle
    ing_words = verb_to_ing(lemma)
    noun_forms = verb_to_noun(lemma)
    adj_forms = noun_to_adj(lemma)
    hardcoded_words = ing_words + noun_forms + adj_forms
    related_words = [f" {s}" for s in set(related_words + hardcoded_words) - {None}]
    
    # Filter for the words which are real words in our tokenizer
    toks = model.to_tokens(related_words, prepend_bos=False)
    related_words = [word[1:] for (word, tok) in zip(related_words, toks) if tok[1] == model.tokenizer.bos_token_id]
    
    # Get forced words (which I won't be checking for if they're tokens, e.g. " Saskatchewans")
    forced_plural = lemma + "s"
    forced_words = [forced_plural]
    
    return [word] + forced_words + related_words


# %%
