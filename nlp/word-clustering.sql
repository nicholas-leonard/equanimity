﻿--DROP TABLE bw.sentence_similarity
--DROP SEQUENCE bw.word_pos_seq;
CREATE SEQUENCE bw.word_pos_seq MINVALUE 1;
CREATE TABLE bw.sentence_word (
	word_pos	INT4 DEFAULT nextval('bw.word_pos_seq'),
	sentence_id	INT4,
	word_id		INT4
);
BEGIN;
INSERT INTO bw.sentence_word (sentence_id, word_id) (
	SELECT sentence_id, unnest(sentence_words) AS word_id
	FROM bw.train_sentence
);--829250940 rows affected
COMMIT; BEGIN;
CREATE INDEX sentence_word_word_id ON bw.sentence_word (word_id);
COMMIT; BEGIN;
CREATE UNIQUE INDEX sentence_word_word_pos ON bw.sentence_word (word_pos);
COMMIT;

SELECT COUNT(*) FROM bw.train_sentence AS a

--idf
--DROP TABLE bw.word_idf2;
CREATE TABLE bw.word_idf2 (
	word_id		INT4,
	word_idf	FLOAT8,
	PRIMARY KEY (word_id)
);

--DROP FUNCTION bw.select_df( word_id INT4) ;
CREATE OR REPLACE FUNCTION bw.select_df( word_id INT4) RETURNS FLOAT8 AS $$
	SELECT COUNT(*)::FLOAT8
	FROM	(
		SELECT DISTINCT sentence_id 
		FROM bw.sentence_word AS b 
		WHERE b.word_id = $1
		) AS a
$$ LANGUAGE 'SQL';

INSERT INTO bw.word_idf2 (word_id, word_idf) (
	SELECT word_id, log(1+(30301028/bw.select_df(a.word_id)))
	FROM bw.word_count AS a
);--793471 rows affected


-- 2. Create a set of context words with counts for each class (word)
-- for building similarity arrows:
--DROP TABLE bw.word_context;
CREATE TABLE bw.word_context (
   target_word    INT4,
   context_word   INT4,
   tfidf          FLOAT8
);

--DROP FUNCTION bw.select_word_context(word_id INT4, context_size INT2) ;
CREATE OR REPLACE FUNCTION bw.select_word_context(word_id INT4, context_size INT2) 
RETURNS TABLE (word_id INT4, sentence_a INT4, sentence_b INT4) AS $$
	SELECT word_id, a.sentence_id, b.sentence_id
	FROM    (
		SELECT word_pos, sentence_id
		FROM bw.sentence_word
		WHERE word_id = $1
		) AS a, bw.sentence_word AS b
	WHERE b.word_pos BETWEEN a.word_pos-$2 AND a.word_pos-1
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION bw.insert_word_contexts(word_id INT4, context_size INT2) RETURNS VOID AS $$
INSERT INTO bw.word_context (target_word, context_word, tfidf) (
	SELECT $1, a.word_id, count*word_idf AS tfidf
	FROM	(
		SELECT word_id, COUNT(*)
		FROM bw.select_word_context($1, $2) AS a
		WHERE sentence_a = sentence_b
		GROUP BY word_id
		) AS a, bw.word_idf2 AS b
	WHERE a.word_id = b.word_id 
	AND (SELECT c.word_id FROM bw.stop_word AS c WHERE c.word_id = a.word_id LIMIT 1) IS NULL 
	ORDER BY tfidf DESC
	LIMIT 700
); $$ LANGUAGE SQL;

python parallel_sql.py "SELECT word_id FROM bw.word_count" "SELECT bw.insert_word_contexts(%s, 10::INT2)" 8


CREATE INDEX word_context_target_word ON bw.word_context (target_word);

SELECT DISTINCT target_word FROM bw.word_context;

--DROP TABLE bw.word_context_c;
CREATE TABLE bw.word_context_c (
	target_word    INT4,
	context_word   INT4,
	tfidf          FLOAT8
);
INSERT INTO bw.word_context_c (target_word, context_word, tfidf) (
	SELECT target_word, context_word, tfidf
	FROM bw.word_context
	ORDER BY context_word
);
CREATE INDEX word_context_context_word ON bw.word_context_c (context_word);

--DROP TABLE bw.word_idf_norm;
CREATE TABLE bw.word_idf_norm (
	word_id		INT4,
	word_idf_norm 	FLOAT8,
	PRIMARY KEY (word_id)
);
INSERT INTO bw.word_idf_norm (word_id, word_idf_norm) (
	SELECT target_word, sqrt(SUM(power(tfidf,2)))::FLOAT8 AS norm
	FROM bw.word_context AS c
	GROUP BY target_word
);

--DROP TABLE bw.word_similarity;
CREATE TABLE bw.word_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	
--http://en.wikipedia.org/wiki/Cosine_similarity
CREATE OR REPLACE FUNCTION bw.build_word_similarity_graph ( cluster_key INT4 ) RETURNS VOID AS $$
INSERT INTO bw.word_similarity (tail, head, similarity) (
        SELECT  $1,
                a.target_word,
                dot_product/(c.word_idf_norm * b.word_idf_norm) AS similarity                
        FROM    (
                SELECT target_word, dot_product
                FROM    (
                        SELECT b.target_word, SUM(a.tfidf*b.tfidf)::FLOAT8 AS dot_product
                        FROM bw.word_context AS a, bw.word_context_c AS b
                        WHERE a.target_word = $1 AND a.context_word = b.context_word 
                        GROUP BY b.target_word
                        ) AS a
                WHERE target_word != $1
                ) AS a, bw.word_idf_norm AS b, bw.word_idf_norm AS c
        WHERE b.word_id = $1 AND c.word_id = a.target_word
        ORDER BY similarity DESC
        LIMIT 700
); $$ LANGUAGE 'SQL';

python parallel_sql.py "SELECT word_id FROM bw.word_count" "SELECT bw.build_word_similarity_graph (%s);" 8

CREATE INDEX cluster5_similarity_tail ON bw.cluster5_similarity (tail);
