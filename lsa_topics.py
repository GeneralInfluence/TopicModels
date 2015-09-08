

from gensim import corpora, models, matutils
import numpy as np

with open(os.path.join(thisdir,"stopwords.txt")) as handle:
    SW = [x[:-1] for x in handle.readlines()]

WORD = re.compile("(([A-Za-z]\.)+)|([0-9]{1,2}:[0-9][0-9])|(([A-za-z]+\'){0,1}\w+(\-\w+){0,1}(\'[a-z]+){0,1})")
def tokenize(text):
    for match in WORD.finditer(text):
        yield match.group()

def load_wikipedia_model():
    print "loading wikipedia dictionary and model..."
    dictionary = pickle.load(open(os.path.join(THISDIR, "lsi-models/wiki_en_dict.pickle")))
    lsi = pickle.load(open(os.path.join(THISDIR, "lsi-models/wiki_en_lsi_200.pickle")))
    return dictionary, lsi



# Wayne: You've already sanitized things. I copy this code for reference only,
#   each document here is what we've referred to as sentences.
documents = [utils.sanitize(row[self.textcol]) for row in self.original_rows]

# Given the documents are cleaned up sentences, we tokenize each document/sentence
#   into a list of lowercase strings
texts = [[word.lower() for word in tokenize(document) if word not in SW] for document in documents]


dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

print "performing lsa..."
# How many topics do the Baltimore emails have? Typically a value from 50-500
#   We chose 200 & 400 dimensions, thus the wikipedia models labeled as 200 and 400 respectively.
numtopics = 200
lsi = models.lsimodel.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=numtopics)
lsi = load_wikipedia_model() # For once you've created your own lsi, just load it.


transformed = matutils.corpus2dense(lsi[corpus_tfidf], len(lsi.projection.s)).T
# normalize the vectors because only the vector orientation represents semantics
transformed_norms = np.sum(transformed**2,axis=-1)**(1./2)
# avoid dividing by zero
transformed_norms[ transformed_norms==0] = 1
transformed = transformed / transformed_norms.reshape(len(transformed_norms),1)


cosine_matrix = 1-transformed.dot(transformed.T)


### NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES NOTES
# STOPWORDS.TXT
a
about
above
after
again
against
all
am
an
and
any
are
aren't
as
at
be
because
been
before
being
below
between
both
but
by
can't
cannot
could
couldn't
did
didn't
do
does
doesn't
doing
don't
down
during
each
few
for
from
further
had
hadn't
has
hasn't
have
haven't
having
he
he'd
he'll
he's
her
here
here's
hers
herself
him
himself
his
how
how's
i
i'd
i'll
i'm
i've
if
in
into
is
isn't
it
it's
its
itself
let's
me
more
most
mustn't
my
myself
no
nor
not
of
off
on
once
only
or
other
ought
our
ours
ourselves
out
over
own
same
shan't
she
she'd
she'll
she's
should
shouldn't
so
some
such
than
that
that's
the
their
theirs
them
themselves
then
there
there's
these
they
they'd
they'll
they're
they've
this
those
through
to
too
under
until
up
very
was
wasn't
we
we'd
we'll
we're
we've
were
weren't
what
what's
when
when's
where
where's
which
while
who
who's
whom
why
why's
with
won't
would
wouldn't
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves
oh
omg
ooh
oooh
ooooh
oooooh
ooooooh
oooooooh
really
actually
wow
already