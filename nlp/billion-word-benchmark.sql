--DROP SCHEMA bw CASCADE;
CREATE SCHEMA bw; --one Billion-Word benchmark
--todo : train, valid, test

-- first parse
CREATE SEQUENCE bw.word_id_seq2 MINVALUE 1;
CREATE TABLE bw.word_count (
	word_id		INT4 DEFAULT nextval('bw.word_id_seq2'),
	word_str	VARCHAR,
	word_count	INT4,
	PRIMARY KEY(word_id),
	UNIQUE (word_str)
);

INSERT INTO bw.word_count (word_str, word_count) VALUES ('<S>', 30301028), ('</S>', 30301028);


-- second parse
--DROP SEQUENCE bw.word_pos_seq CASCADE; DROP TABLE bw.sentence_word;
CREATE SEQUENCE bw.word_pos_seq MINVALUE 1;
CREATE TABLE bw.train_sentence (
	word_pos	INT4 DEFAULT nextval('bw.word_pos_seq'),
	sentence_id	INT4,
	word_id		INT4,
	PRIMARY KEY (word_pos)
);
CREATE TABLE bw.valid_sentence (
	word_pos	INT4 DEFAULT nextval('bw.word_pos_seq'),
	sentence_id	INT4,
	word_id		INT4,
	PRIMARY KEY (word_pos)
);
CREATE TABLE bw.test_sentence (
	word_pos	INT4 DEFAULT nextval('bw.word_pos_seq'),
	sentence_id	INT4,
	word_id		INT4,
	PRIMARY KEY (word_pos)
);

CREATE INDEX train_sentence_id ON bw.train_sentence (sentence_id);
CREATE INDEX train_word_id ON bw.train_sentence (word_id);

INSERT INTO bw.sentence_word (sentence_id, word_id) (SELECT 0, unnest(ARRAY[100,200,300]::INT4[])) 
SELECT * FRoM bw.sentence_word;

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
-- for building similarity arrows:
CREATE TABLE bw.context (
   target_word_id       INT4,
   context_word_id       INT4,
   count             INT4,
   PRIMARY KEY (target_word_id, context_word_id)
);

CREATE OR REPLACE FUNCTION bw.get_word_contexts(word_id INT4, context_size INT2) 
RETURNS INT4[] AS $$
   SELECT word_id, COUNT(*)
   FROM  (
      SELECT word_pos, sentence_id
      FROM bw.sentence_words
      WHERE word_id = $1
   ) AS a, bw.sentence_words AS b
   WHERE b.word_pos BETWEEN a.word_pos-$2 AND a.word_pos-1
   AND a.sentence_id = b.sentence_id
   GROUP BY word_id
   ORDER BY count DESC
$$ LANGUAGE SQL;

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

