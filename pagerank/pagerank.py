import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    probability = {}
    # check if there are outgoing links
    if len(corpus[page]) == 0:
        for s_page in corpus:
            probability[s_page] = 1 / len(corpus.keys())
    else:

        # let's add pages to our dict
        for s_page in corpus:
            probability[s_page] = (1 - damping_factor) * 1 / len(corpus.keys())

        outgoing_links = len(corpus[page])
        for ext_page in corpus[page]:
            probability[ext_page] += damping_factor * 1 / outgoing_links

    # print(page, probability)
    return probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # pre-count every probability for every page, because they won't change but take some time to process every time
    probability_base = {}
    for page in corpus:
        probability_base[page] = transition_model(corpus, page, damping_factor)

    # dict representing how many times surfer visited a page
    count_base = {}
    for page in corpus:
        count_base[page] = 0

    counter = n
    current_page = random.choice(list(corpus.keys()))
    count_base[current_page] += 1
    counter -= 1
    while counter > 0:
        current_page = random.choices(list(corpus.keys()), weights=list(probability_base[current_page].values()))[0]
        count_base[current_page] += 1
        counter -= 1

    for page in count_base:
        count_base[page] /= n

    return count_base


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # set up probabilities
    probability = {}
    for page in corpus:
        probability[page] = 1 / len(corpus.keys())

    iteration = True
    while iteration:
        iteration = False
        new_probability = {}
        for page_1 in corpus:
            new_probability[page_1] = (1 - damping_factor) / len(corpus.keys())
            for page_2 in corpus:
                if page_2 != page_1 and page_1 in corpus[page_2]:
                    new_probability[page_1] += damping_factor * probability[page_2] / len(corpus[page_2])
            if abs(probability[page_1] - new_probability[page_1]) > 0.001:
                iteration = True
        probability = new_probability

    return probability


if __name__ == "__main__":
    main()
