SELECT a.sentence_id, b.word_str
FROM bw.sentence_bag_s AS a, bw.word_count AS b
WHERE a.word_id = b.word_id AND (sentence_id = 1 OR sentence_id = 680457 OR sentence_id = 622357 OR sentence_id = 718412 OR sentence_id = 326372 OR sentence_id = 248225)
ORDER BY sentence_id ASC

SELECT MAX(sentence_id), COUNT(*) FROM bw.train_sentence;

SELECT SUM(array_upper(sentence_words, 1)-1) FROM bw.train_sentence AS a WHERE sentence_id < 38000

SELECT word_id, word_str, word_count FROM bw.word_count ORDER BY word_count DESC LIMIT 100
SELECT MAX(word_id), MIN(word_id), COUNT(*) FROM bw.word_count

SELECT a.cluster_key, a.density, s_str
FROM	(
	SELECT a.cluster_key, a.sum, a.density, a.sentence_id, array_agg(b.word_str) AS s_str
	FROM	(
		SELECT a.cluster_key, c.sentence_id, a.sum, b.density, unnest(c.sentence_words) AS word_id, generate_series(1,array_upper(c.sentence_words, 1),1)
		FROM	(
			SELECT a.cluster_key, SUM(a.density)
			FROM public.itemclusters AS a
			GROUP BY a.cluster_key
			ORDER BY sum DESC 
			--OFFSET 10000 
			LIMIT 100
			) AS a, public.itemclusters AS b, bw.train_sentence AS c
		WHERE b.cluster_key = a.cluster_key AND c.sentence_id = b.item_key
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id
	GROUP BY a.cluster_key, a.sum, a.density, a.sentence_id
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.density DESC

SELECT now(), MIN(density), MAX(density), AVG(density), SUM(density) FROM public.itemclusters;
/*
            now(),          MIN(density), MAX(density),   AVG(density),   SUM(density)
"2014-03-20 23:54:59.710361-04";1e-16;1.16278526970974;0.163805230469986;181959.272747284
"2014-03-21 00:18:07.886911-04";1e-16;1.16278526970974;0.166448847278408;184895.873675732
"2014-03-21 00:30:48.09286-04";1e-16;1.16278526970974;0.168515577950591;187191.653908121
"2014-03-21 00:53:53.63846-04";1e-16;1.16278526970974;0.172188308930854;191271.422644734
"2014-03-21 09:25:33.233244-04";1e-16;1.16278526970974;0.314205694513418;349028.169019257
"2014-03-21 13:10:58.913406-04";1e-16;1.48666224703685;0.420995068245878;467652.688674364
"2014-03-22 00:24:58.712561-04";1e-16;3.51389635469039;0.8905936699262;989295.494583111
"2014-03-22 00:37:44.277788-04";1e-16;3.51389635469039;0.907359028503606;1007918.90755558
"2014-03-22 02:01:05.506585-04";1e-16;3.56378090153832;1.02269141083084;1136033.23181899
"2014-03-22 10:26:55.971605-04";1e-16;6.12365713431746;1.40286055478212;1558335.38148695
"2014-03-22 21:24:00.897714-04";1e-16;6.92167925138953;1.53828965100194;1708773.67815353
"2014-03-23 01:48:27.084485-04";1e-16;7.41260336775986;1.57696802496858;1751738.66027178
"2014-03-23 22:20:20.626472-04";1e-16;8.35835535598301;1.6673916074904;1852183.61717374
"2014-03-23 22:37:31.333598-04";1e-16;8.35835535598301;1.66839341894663;1853296.45638823

words
"2014-03-28 16:05:33.170579-04";1e-16;1;0.00134632532831147;1055.03303395267
"2014-03-28 16:13:38.606203-04";1e-16;1;0.017996500020248;14102.7592793671
"2014-03-29 11:22:31.372434-04";1e-16;8.47072033622108;0.92217552845428;722652.708942383
"2014-03-29 12:16:59.851281-04";1e-16;8.61556075006084;0.979858443499282;767855.290805334
"2014-03-29 17:55:04.052022-04";1e-16;8.83678667885449;1.20447788413099;943875.844642522
"2014-03-31 13:27:38.864168-04";1e-16;9;1.70038336885958;1,332,486.72278975
"2014-04-01 16:09:32.944039-04";1e-16;9;1.75897439637015;1,378,400.93699711
"2014-04-02 14:05:55.100312-04";1e-16;9;1.78401311106235;1,398,022.25033979
*/

SELECT a.cluster_key, s_str, count
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT a.cluster_key, SUM(a.density)
			FROM bw.itemclusters AS a
			GROUP BY a.cluster_key
			ORDER BY sum DESC 
			--OFFSET 10000 
			LIMIT 100
			) AS a, public.itemclusters AS b, bw.sentence_bag_s AS c
		WHERE b.cluster_key = a.cluster_key AND c.sentence_id = b.item_key
		GROUP BY a.cluster_key, a.sum, word_id
		ORDER BY count DESC
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 2
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.count DESC

-- mean max count of intra-cluster shared words

SELECT now(), avg_max, avg_c_count, avg_w_count, count
FROM	(
	SELECT AVG(max) AS avg_max, AVG(count) AS avg_c_count, AVG(avg) AS avg_w_count, COUNT(*) AS count
	FROM	(
		SELECT a.cluster_key, MAX(count), AVG(count), COUNT(*)
		FROM	(
			SELECT a.cluster_key, word_id, COUNT(*)
			FROM	(
				SELECT a.cluster_key, SUM(a.density)
				FROM public.itemclusters AS a
				GROUP BY a.cluster_key
				ORDER BY sum DESC 
				) AS a, public.itemclusters AS b, bw.sentence_bag_s AS c
			WHERE b.cluster_key = a.cluster_key AND c.sentence_id = b.item_key
			GROUP BY a.cluster_key, word_id
			ORDER BY count DESC
			) AS a
		WHERE a.count > 1
		GROUP BY a.cluster_key
		) AS a
	) AS a

/* now(),                            avg_max,          avg_c_count,     count
3
"2014-03-21 13:52:35.441314-04";4.7733705564102825;1.7406441358370607;98147
"2014-03-21 15:20:22.531831-04";5.1599590469099032;2.0272803425167535;107440
"2014-03-22 00:17:00.394401-04";8.1290331292761973;2.2732535109830753;111080
   now(),                            avg_max,          avg_c_count,       avg_w_count,   count
"2014-03-22 00:44:56.994154-04";8.3704110476944960;2.2395077510307701;7.3980887756063911;111082
"2014-03-22 02:01:47.718675-04";8.9540339568967069;2.1662105471633568;7.9191011500984918;111082
"2014-03-22 10:22:38.842706-04";9.8293332853207540;2.2208368592571254;8.6426498382016498;111082
1
"2014-03-22 10:24:18.783845-04";9.8297563961757981;9.0558506328658108;3.9192668227864475;111082
"2014-03-22 21:21:10.067412-04";9.8923678003636953;9.3721845123422337;3.9723368425094516;111082
"2014-03-23 01:47:32.660693-04";9.8975441565690211;9.4239390720368737;4.0000211504921609;111082
"2014-03-23 22:36:22.910997-04";9.9055562557390036;9.4926990871608361;4.0600790515801552;111082
*/

-- inequality graph

SELECT cluster_key, sum
FROM	(
	SELECT row_number() OVER ( ORDER BY sum DESC), *
	FROM	(
		SELECT cluster_key, SUM(density) 
		FROM public.itemclusters
		GROUP BY cluster_key
		) AS a
	) AS a
WHERE row_number % 1000 = 0
ORDER BY sum DESC

-- cluster 4

SELECT a.cluster_key, s_str, count
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT a.cluster_key, SUM(a.density)
			FROM public.itemclusters AS a
			GROUP BY a.cluster_key
			ORDER BY sum DESC 
			--OFFSET 10000 
			LIMIT 100
			) AS a, public.itemclusters AS b, bw.sentence_bag_s AS c, bw.cluster5 AS d
		WHERE b.cluster_key = a.cluster_key AND b.item_key = d.cluster_key AND c.sentence_id = d.item_key 
		GROUP BY a.cluster_key, a.sum, word_id
		ORDER BY count DESC
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 2
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.count DESC

SELECT a.cluster_key, s_str, count
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT a.cluster_key, SUM(a.density)
			FROM bw.cluster4  AS a
			WHERE cluster_key = 4264
			GROUP BY a.cluster_key
			) AS a, bw.cluster4 AS b, bw.sentence_bag_s AS c, bw.cluster5 AS d
		WHERE b.cluster_key = a.cluster_key AND b.item_key = d.cluster_key AND c.sentence_id = d.item_key 
		GROUP BY a.cluster_key, a.sum, word_id
		ORDER BY count DESC
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 2
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.count DESC;


-- cluster 3

SELECT a.cluster_key, s_str, count
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT a.cluster_key, SUM(a.density)
			FROM bw.cluster3 AS a
			GROUP BY a.cluster_key
			ORDER BY sum DESC 
			--OFFSET 10000 
			LIMIT 100
			) AS a, bw.cluster3 AS b, bw.sentence_bag_s AS c, bw.cluster5 AS d, bw.cluster4 AS e
		WHERE b.cluster_key = a.cluster_key AND b.item_key = e.cluster_key 
		AND e.item_key = d.cluster_key AND c.sentence_id = d.item_key 
		GROUP BY a.cluster_key, a.sum, word_id
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 30
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.count DESC

SELECT * FROM public.itemclusters WHERE cluster_key = 684 ORDER BY density DESC


SELECT a.cluster_key, s_str, count
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT DISTINCT a.cluster_key, e.cluster_key AS cluster4_key, a.sum, word_id
			FROM	(
				SELECT a.cluster_key, SUM(a.density)
				FROM public.itemclusters AS a
				GROUP BY a.cluster_key
				ORDER BY sum DESC 
				--OFFSET 10000 
				LIMIT 100
				) AS a, public.itemclusters AS b, bw.sentence_bag_s AS c, bw.cluster5 AS d, bw.cluster4 AS e
			WHERE b.cluster_key = a.cluster_key AND b.item_key = e.cluster_key 
			AND e.item_key = d.cluster_key AND c.sentence_id = d.item_key 
			) AS a
		GROUP BY a.cluster_key, a.sum, word_id
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 5
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.count DESC;

SELECT s_str, COUNT(*)
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT DISTINCT a.cluster_key, e.cluster_key AS cluster4_key, a.sum, word_id
			FROM	(
				SELECT a.cluster_key, SUM(a.density)
				FROM public.itemclusters AS a
				GROUP BY a.cluster_key
				ORDER BY sum DESC 
				--OFFSET 10000 
				LIMIT 100
				) AS a, public.itemclusters AS b, bw.sentence_bag_s AS c, bw.cluster5 AS d, bw.cluster4 AS e
			WHERE b.cluster_key = a.cluster_key AND b.item_key = e.cluster_key 
			AND e.item_key = d.cluster_key AND c.sentence_id = d.item_key 
			) AS a
		GROUP BY a.cluster_key, a.sum, word_id
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 5
	) AS a
GROUP BY s_str
ORDER BY count DESC;

-- cluster 2

SELECT a.cluster_key, s_str, count
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT a.cluster_key, SUM(a.density)
			FROM bw.cluster2 AS a
			GROUP BY a.cluster_key
			ORDER BY sum DESC 
			--OFFSET 10000 
			LIMIT 10
			) AS a, bw.cluster2 AS b, bw.sentence_bag_s AS c, bw.cluster5 AS d, bw.cluster4 AS e, bw.cluster3 AS f
		WHERE b.cluster_key = a.cluster_key AND b.item_key = f.cluster_key AND f.item_key = e.cluster_key
		AND e.item_key = d.cluster_key AND c.sentence_id = d.item_key 
		GROUP BY a.cluster_key, a.sum, word_id
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 30
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.count DESC

-- cluster 1

SELECT a.cluster_key, s_str, count
FROM	(
	SELECT a.cluster_key, a.sum, b.word_str AS s_str, count
	FROM	(
		SELECT a.cluster_key, a.sum, word_id, COUNT(*)
		FROM	(
			SELECT a.cluster_key, SUM(a.density)
			FROM bw.cluster1 AS a
			GROUP BY a.cluster_key
			ORDER BY sum DESC 
			--OFFSET 10000 
			LIMIT 10
			) AS a, bw.cluster1 AS b, bw.sentence_bag_s AS c, bw.cluster5 AS d, bw.cluster4 AS e, bw.cluster3 AS f, bw.cluster2 AS g
		WHERE b.cluster_key = a.cluster_key AND b.item_key = g.cluster_key AND g.item_key = f.cluster_key 
		AND f.item_key = e.cluster_key AND e.item_key = d.cluster_key AND c.sentence_id = d.item_key 
		GROUP BY a.cluster_key, a.sum, word_id
		) AS a, bw.word_count AS b
	WHERE a.word_id = b.word_id AND a.count > 300
	) AS a
ORDER BY a.sum DESC, a.cluster_key ASC, a.count DESC;

--word cluster5

SELECT a.cluster_key, a.sum, b.word_str AS s_str, density
FROM	(
	SELECT a.cluster_key, a.sum, b.item_key, b.density
	FROM	(
		SELECT a.cluster_key, SUM(a.density)
		FROM bw.word_cluster5 AS a
		GROUP BY a.cluster_key
		ORDER BY sum DESC 
		OFFSET 10000 
		LIMIT 100
		) AS a, bw.word_cluster5 AS b
	WHERE b.cluster_key = a.cluster_key
	) AS a, bw.word_count AS b
WHERE a.item_key = b.word_id
ORDER BY a.sum DESC, a.cluster_key ASC, a.density DESC

SELECT a.cluster_key, array_agg(a.s_str)
FROM(
SELECT a.cluster_key, a.sum, b.word_str AS s_str, density
FROM	(
	SELECT a.cluster_key, a.sum, b.item_key, b.density
	FROM	(
		SELECT a.cluster_key, SUM(a.density)
		FROM bw.word_cluster5 AS a
		GROUP BY a.cluster_key
		ORDER BY sum DESC 
		OFFSET 1000
		LIMIT 100
		) AS a, bw.word_cluster5 AS b
	WHERE b.cluster_key = a.cluster_key
	) AS a, bw.word_count AS b
WHERE a.item_key = b.word_id
ORDER BY a.sum DESC, a.cluster_key ASC, a.density DESC
) AS a
GROUP BY a.cluster_key

--word cluster4

SELECT a.cluster_key, a.sum, b.word_str AS s_str, a.density*c.density AS density
FROM	(
	SELECT a.cluster_key, a.sum, b.item_key, b.density
	FROM	(
		SELECT a.cluster_key, SUM(a.density)
		FROM bw.word_cluster4 AS a
		GROUP BY a.cluster_key
		ORDER BY sum DESC 
		--OFFSET 10000 
		LIMIT 100
		) AS a, bw.word_cluster4 AS b
	WHERE b.cluster_key = a.cluster_key
	) AS a, bw.word_count AS b, bw.word_cluster5 AS c
WHERE a.item_key = c.cluster_key AND c.item_key = b.word_id
ORDER BY a.sum DESC, a.cluster_key ASC, a.density*c.density DESC

--word cluster3

SELECT a.cluster_key, a.sum, b.word_str AS s_str, a.density*c.density*d.density AS density
FROM	(
	SELECT a.cluster_key, a.sum, b.item_key, b.density
	FROM	(
		SELECT a.cluster_key, SUM(a.density)
		FROM bw.word_cluster3 AS a
		GROUP BY a.cluster_key
		ORDER BY sum DESC 
		--OFFSET 10000 
		LIMIT 100
		) AS a, bw.word_cluster3 AS b
	WHERE b.cluster_key = a.cluster_key
	) AS a, bw.word_count AS b, bw.word_cluster5 AS c, bw.word_cluster4 AS d
WHERE c.item_key = b.word_id AND a.item_key = d.cluster_key AND d.item_key = c.cluster_key 
ORDER BY a.sum DESC, a.cluster_key ASC, a.density*c.density*d.density DESC

--word cluster2

SELECT a.cluster_key, a.sum, b.word_str AS s_str, a.density*c.density*d.density*e.density AS density
FROM	(
	SELECT a.cluster_key, a.sum, b.item_key, b.density
	FROM	(
		SELECT a.cluster_key, SUM(a.density)
		FROM bw.word_cluster2 AS a
		GROUP BY a.cluster_key
		ORDER BY sum DESC 
		--OFFSET 10000 
		LIMIT 10
		) AS a, bw.word_cluster2 AS b
	WHERE b.cluster_key = a.cluster_key
	) AS a, bw.word_count AS b, bw.word_cluster5 AS c, bw.word_cluster4 AS d, bw.word_cluster3 AS e
WHERE c.item_key = b.word_id AND a.item_key = e.cluster_key AND e.item_key = d.cluster_key 
AND d.item_key = c.cluster_key AND  a.density*c.density*d.density*e.density > 3500
ORDER BY a.sum DESC, a.cluster_key ASC, a.density*c.density*d.density*e.density DESC

--word cluster1

SELECT a.cluster_key, a.sum, b.word_str AS s_str, a.density*c.density*d.density*e.density*f.density AS density
FROM	(
	SELECT a.cluster_key, a.sum, b.item_key, b.density
	FROM	(
		SELECT a.cluster_key, SUM(a.density)
		FROM bw.word_cluster1 AS a
		GROUP BY a.cluster_key
		ORDER BY sum DESC 
		--OFFSET 10000 
		LIMIT 10
		) AS a, bw.word_cluster1 AS b
	WHERE b.cluster_key = a.cluster_key
	) AS a, bw.word_count AS b, bw.word_cluster5 AS c, bw.word_cluster4 AS d, bw.word_cluster3 AS e, bw.word_cluster2 AS f
WHERE c.item_key = b.word_id AND a.item_key = f.cluster_key AND f.item_key = e.cluster_key AND e.item_key = d.cluster_key 
AND d.item_key = c.cluster_key AND  a.density*c.density*d.density*e.density*f.density > 3500
ORDER BY a.sum DESC, a.cluster_key ASC, a.density*c.density*d.density*e.density*f.density DESC