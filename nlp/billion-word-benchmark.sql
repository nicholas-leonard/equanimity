CREATE SCHEMA bw; --one Billion-Word benchmark
--todo : train, valid, test

CREATE TABLE bw.shards (
	shard_path 	VARCHAR,
	shard_file	VARCHAR,
	shard_id   	INT2,
	PRIMARY KEY (shard_id)
);

CREATE TABLE bw.shard_word_counts (
	shard_id	INT2,
	word_str 	VARCHAR,
	word_count 	INT4,
	PRIMARY KEY (shard_id, word_str)
);
CREATE INDEX shard_word_counts_word_str ON bw.shard_word_counts (word_str);

CREATE SEQUENCE bw.word_id_seq MINVALUE 1;
CREATE TABLE bw.word_counts (
	word_id		INT4 DEFAULT nextval('bw.word_id_seq'),
	word_str	VARCHAR,
	word_count	INT4,
	PRIMARY KEY(word_id),
	UNIQUE (word_str)
);

CREATE TABLE bw.unknown_counts (
	word_str	VARCHAR,
	word_count	INT4,
	PRIMARY KEY(word_str)
);

--option 1 (like a file : efficient representation)
CREATE TABLE bw.sentence_words (
	sentence_id	INT4,
	word_ids	INT4[],
);

--option 2 (maximum flexibility)
CREATE SEQUENCE bw.word_pos_seq MINVALUE 1;
CREATE TABLE bw.sentence_words (
	word_pos	INT4 DEFAULT nextval('bw.word_pos_seq'),
	sentence_id	INT4,
	word_id		INT4,
	PRIMARY KEY (word_pos)
);
CREATE INDEX sentence_words_sentence_id ON bw.sentence_words (sentence_id);
CREATE INDEX sentence_words_word_id ON bw.sentence_words (word_id);

CREATE TABLE bw.expert_examples(
	expert_id	INT2,
	word_pos	INT4,
	PRIMARY KEY (word_pos)
)

CREATE TABLE bw.expert_examples(
	expert_id	INT2,
	sentence_id	INT4,
	word_idx	INT2,
	PRIMARY KEY (sentence_id, word_idx)
);
CREATE INDEX expert_idx...;

--Requirements
-- 1. Select batch of examples by word_pos for n context words
CREATE OR REPLACE FUNCTION bw.get_example(word_positions INT4, context_size INT2) RETURNS INT4[] AS $$
	SELECT array_agg(word_id)
	FROM 	(
		SELECT word_id
		FROM bw.sentence_words
		WHERE word_pos BETWEEN $1-$2 AND $1
		AND sentence_id = (SELECT sentence_id FROM bw.sentence_words WHERE word_pos = $1)
		ORDER BY word_pos ASC
	) AS a
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION bw.get_batch(word_positions INT4[], context_size INT2) RETURNS SETOF INT4[] AS $$
	SELECT bw.get_example(unnest($1), $2)
	--FROM (SELECT unnest($1) AS word_pos) AS a
$$ LANGUAGE SQL;

-- 2. Create a set of context words with counts for each class (word)

SELECT 
SELECT unnest(get_examples(word_pos, 5)) AS word
WHERE word_id = $1
