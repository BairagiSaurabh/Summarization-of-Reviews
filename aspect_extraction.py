def apply_extraction(row, nlp):
        """
        This function extracts aspect and its corresponding description from the review by
        applying 7 different rules of pos tagging

        """

        prod_pronouns = ['it', 'this', 'they', 'these']
        review_body = row['Review']
        doc = nlp(review_body)

        rule1_pairs = []
        rule2_pairs = []
        rule3_pairs = []
        rule4_pairs = []
        rule5_pairs = []
        rule6_pairs = []
        rule7_pairs = []

        for token in doc:
            A = "999999"
            M = "999999"
            if token.dep_ == "amod" and not token.is_stop:
                M = token.text
                A = token.head.text

                # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
                M_children = token.children
                for child_m in M_children:
                    if (child_m.dep_ == "advmod"):
                        M_hash = child_m.text
                        M = M_hash + " " + M
                        break

                # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
                A_children = token.head.children
                for child_a in A_children:
                    if (child_a.dep_ == "det" and child_a.text == 'no'):
                        neg_prefix = 'not'
                        M = neg_prefix + " " + M
                        break

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict1 = {"noun": A, "adj": M, "rule": 1}
                rule1_pairs.append(dict1)

            # print("--- SPACY : Rule 1 Done ---")

            # -----------------------------------------------------------------------------------------------------------------------------
            # # SECOND RULE OF DEPENDANCY PARSE -
            # # M - Sentiment modifier || A - Aspect
            # Direct Object - A is a child of something with relationship of nsubj, while
            # M is a child of the same something with relationship of dobj
            # Assumption - A verb will have only one NSUBJ and DOBJ
            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if ((child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop):
                    M = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict2 = {"noun": A, "adj": M, "rule": 2}
                rule2_pairs.append(dict2)

            # print("--- SPACY : Rule 2 Done ---")
            # -----------------------------------------------------------------------------------------------------------------------------

            ## THIRD RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## Adjectival Complement - A is a child of something with relationship of nsubj, while
            ## M is a child of the same something with relationship of acomp
            ## Assumption - A verb will have only one NSUBJ and DOBJ
            ## "The sound of the speakers would be better. The sound of the speakers could be better" - handled using AUX dependency

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "acomp" and not child.is_stop):
                    M = child.text

                # example - 'this could have been better' -> (this, not better)
                if (child.dep_ == "aux" and child.tag_ == "MD"):
                    neg_prefix = "not"
                    add_neg_pfx = True

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M
                # check_spelling(child.text)

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict3 = {"noun": A, "adj": M, "rule": 3}
                rule3_pairs.append(dict3)
                # rule3_pairs.append((A, M, sid.polarity_scores(M)['compound'],3))
            # print("--- SPACY : Rule 3 Done ---")
            # ------------------------------------------------------------------------------------------------------------------------------

            ## FOURTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect

            # Adverbial modifier to a passive verb - A is a child of something with relationship of nsubjpass, while
            # M is a child of the same something with relationship of advmod

            # Assumption - A verb will have only one NSUBJ and DOBJ

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if ((child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "advmod" and not child.is_stop):
                    M = child.text
                    M_children = child.children
                    for child_m in M_children:
                        if (child_m.dep_ == "advmod"):
                            M_hash = child_m.text
                            M = M_hash + " " + child.text
                            break
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict4 = {"noun": A, "adj": M, "rule": 4}
                rule4_pairs.append(dict4)
                # rule4_pairs.append((A, M,sid.polarity_scores(M)['compound'],4)) # )

            # print("--- SPACY : Rule 4 Done ---")
            # ------------------------------------------------------------------------------------------------------------------------------

            ## FIFTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect

            # Complement of a copular verb - A is a child of M with relationship of nsubj, while
            # M has a child with relationship of cop

            # Assumption - A verb will have only one NSUBJ and DOBJ

            children = token.children
            A = "999999"
            buf_var = "999999"
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "cop" and not child.is_stop):
                    buf_var = child.text
                    # check_spelling(child.text)

            if (A != "999999" and buf_var != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict5 = {"noun": A, "adj": token.text, "rule": 5}
                rule5_pairs.append(dict5)
                # rule5_pairs.append((A, token.text,sid.polarity_scores(token.text)['compound'],5))

            # print("--- SPACY : Rule 5 Done ---")
            # ------------------------------------------------------------------------------------------------------------------------------

            ## SIXTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## Example - "It ok", "ok" is INTJ (interjections like bravo, great etc)

            children = token.children
            A = "999999"
            M = "999999"
            if (token.pos_ == "INTJ" and not token.is_stop):
                for child in children:
                    if (child.dep_ == "nsubj" and not child.is_stop):
                        A = child.text
                        M = token.text
                        # check_spelling(child.text)

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict6 = {"noun": A, "adj": M, "rule": 6}
                rule6_pairs.append(dict6)

                # rule6_pairs.append((A, M,sid.polarity_scores(M)['compound'],6))

            # print("--- SPACY : Rule 6 Done ---")

            # ------------------------------------------------------------------------------------------------------------------------------

            ## SEVENTH RULE OF DEPENDANCY PARSE -
            ## M - Sentiment modifier || A - Aspect
            ## ATTR - link between a verb like 'be/seem/appear' and its complement
            ## Example: 'this is garbage' -> (this, garbage)

            children = token.children
            A = "999999"
            M = "999999"
            add_neg_pfx = False
            for child in children:
                if (child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    # check_spelling(child.text)

                if ((child.dep_ == "attr") and not child.is_stop):
                    M = child.text
                    # check_spelling(child.text)

                if (child.dep_ == "neg"):
                    neg_prefix = child.text
                    add_neg_pfx = True

            if (add_neg_pfx and M != "999999"):
                M = neg_prefix + " " + M

            if (A != "999999" and M != "999999"):
                if A in prod_pronouns:
                    A = "product"
                dict7 = {"noun": A, "adj": M, "rule": 7}
                rule7_pairs.append(dict7)
                # rule7_pairs.append((A, M,sid.polarity_scores(M)['compound'],7))

        # print("--- SPACY : All Rules Done ---")

        # ------------------------------------------------------------------------------------------------------------------------------

        aspects = []

        aspects = rule1_pairs + rule2_pairs + rule3_pairs + rule4_pairs + rule5_pairs + rule6_pairs + rule7_pairs

        dic = {"aspect_pairs": aspects}
        return dic