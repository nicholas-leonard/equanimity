CREATE TABLE bw.sentence_bag (
	sentence_id	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
CREATE INDEX sentence_bag_sentence_id ON bw.sentence_bag (sentence_id);
CREATE INDEX sentence_bag_word_id ON bw.sentence_bag (word_id);

--tf
INSERT INTO bw.sentence_bag (sentence_id, word_id, word_freq) (
	SELECT sentence_id, word_id, COUNT(*)
	FROM bw.train_sentence
	WHERE sentence_id BETWEEN 1 AND 1000000
	GROUP BY sentence_id, word_id
);--24,607,658 rows affected

--idf : http://en.wikipedia.org/wiki/Tf%E2%80%93idf
CREATE TABLE bw.word_idf (
	word_id		INT4,
	word_idf	FLOAT8,
	PRIMARY KEY (word_id)
);

CREATE OR REPLACE FUNCTION bw.select_idf( word_id INT4) RETURNS FLOAT8 AS $$
	SELECT COUNT(*)::FLOAT8 FROM bw.sentence_bag AS b WHERE b.word_id = $1
$$ LANGUAGE 'SQL';

INSERT INTO bw.word_idf (word_id, word_idf) (
	SELECT word_id, log(1000000/bw.select_idf(a.word_id))
	FROM 	(
		SELECT DISTINCT word_id 
		FROM bw.sentence_bag
		) AS a
);--302647 rows affected

--tf-idf
UPDATE bw.sentence_bag AS a
SET word_freq = word_freq * word_idf
FROM bw.word_idf AS b
WHERE a.word_id = b.word_id--24607658 rows affected


--optimize
CREATE TABLE bw.sentence_bag_s (
	sentence_id	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.sentence_bag_s (sentence_id, word_id, word_freq) (
	SELECT sentence_id, word_id, word_freq
	FROM bw.sentence_bag
	ORDER BY sentence_id ASC
);
CREATE INDEX sentence_bag_s_idx ON bw.sentence_bag_s (sentence_id);

CREATE TABLE bw.sentence_bag_w (
	sentence_id	INT4,
	word_id		INT4,
	word_freq	FLOAT8
);
INSERT INTO bw.sentence_bag_w (sentence_id, word_id, word_freq) (
	SELECT sentence_id, word_id, word_freq
	FROM bw.sentence_bag
	ORDER BY word_id ASC
);
CREATE INDEX sentence_bag_w_idx ON bw.sentence_bag_w (word_id);

DROP TABLE bw.sentence_bag;

--DROP TABLE bw.sentence_similarity;
CREATE TABLE bw.sentence_similarity (
	tail		INT4,
	head		INT4,
	similarity	FLOAT8
);
	

CREATE OR REPLACE FUNCTION bw.build_sentence_similarity_graph ( sentence_id INT4 ) RETURNS VOID AS $$
INSERT INTO bw.sentence_similarity (tail, head, similarity) (
        SELECT  1,
                sentence_id,
                dot_product
                /(
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.sentence_bag_s AS c
                WHERE c.sentence_id = a.sentence_id
                )
                *
                b.norm
                ) AS similarity                
        FROM    (
                SELECT sentence_id, dot_product
                FROM    (
                        SELECT b.sentence_id, SUM(a.word_freq*b.word_freq)::FLOAT8 AS dot_product
                        FROM bw.sentence_bag_s AS a, bw.sentence_bag_w AS b
                        WHERE a.sentence_id = 1 AND a.word_id = b.word_id 
                        GROUP BY b.sentence_id
                        ) AS a
                WHERE sentence_id != 1
                ) AS a,
                (
                SELECT sqrt(SUM(power(word_freq,2)))::FLOAT8 AS norm
                FROM bw.sentence_bag_s
                WHERE sentence_id = 1
                ) AS b
        ORDER BY similarity DESC
        LIMIT 100
);
$$ LANGUAGE 'SQL';

SELECT 

python parallel_sql.py "SELECT DISTINCT sentence_id FROM bw.sentence_bag" "SELECT bw.build_sentence_similarity_graph (%s);" 8