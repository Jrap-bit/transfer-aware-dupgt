"""Check descriptive-text cleaning and release-safe joins on small examples."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dupgt_cdr.metadata import clean_record, clean_domain, coverage_report, item_key

SYNOPSIS = "A young scientist travels to a distant planet and discovers a hidden civilization while searching for her missing family."


class MetadataTests(unittest.TestCase):
    def test_later_synopsis_survives_author_biography(self):
        row = dict(asin='B1', title='Distant Planet', description=[
            'Jane Smith is a professor of astronomy at the university and she lives in London with her family.', SYNOPSIS], categories=[['Books','Science Fiction']])
        cleaned, diagnostic = clean_record(row, '2014', 'books')
        self.assertEqual(cleaned['description'], SYNOPSIS)
        self.assertNotIn('professor', cleaned['description'])
        self.assertEqual(cleaned['categories'], 'Science Fiction')
        self.assertEqual(diagnostic['discarded_passages']['author_biography'], 1)

    def test_format_and_biography_alone_are_rejected(self):
        for text in ('vhs', '', 'EXCELLENT!!!!!!!!!!!!!!!!',
                     'Jane Smith is a professor of astronomy at the university and she lives in London with her family.',
                     'Jane Smith is a corporate presentation coach and award-winning journalist who spent fifteen years working for several national media outlets.',
                     'Joseph Smithis cofounder of a university of natural medicine and a health writer and medical editor with many years of professional experience.'):
            cleaned, diagnostic = clean_record(dict(asin='B1', title='Title', description=text), '2014', 'books')
            self.assertIsNone(cleaned)
            self.assertFalse(diagnostic['accepted'])

    def test_parent_asin_and_feature_fallback_do_not_copy_outcomes(self):
        row = dict(parent_asin='PARENT', asin='WRONG-CHILD', title='Distant Planet',
            description=[], features=[SYNOPSIS], average_rating=5, rating_number=999,
            bought_together=['PRIVATE'], reviewText='PRIVATE', details={'Genre':'Science fiction','Customer Reviews':'PRIVATE'})
        cleaned, diagnostic = clean_record(row, '2023', 'movies')
        self.assertEqual(cleaned['asin'], 'PARENT')
        self.assertEqual(diagnostic['text_fields'], ['features'])
        self.assertIn('Science fiction', cleaned['categories'])
        self.assertEqual(set(cleaned), {'dataset_release','asin','title','description','categories'})
        self.assertNotIn('PRIVATE', json.dumps(cleaned))
        self.assertIsNone(item_key({'asin':'CHILD'}, '2023'))
        with self.assertRaises(ValueError):
            item_key({'dataset_release':'2014','asin':'PARENT'}, '2023')

    def test_aggregate_rating_text_is_removed_even_inside_a_description(self):
        row = dict(asin='B1',title='Distant Planet', description=[SYNOPSIS,
            'This novel follows an astronaut and has 4.8 out of 5 stars from thousands of readers across the country.'])
        cleaned, diagnostic = clean_record(row,'2014','books')
        self.assertEqual(cleaned['description'],SYNOPSIS)
        self.assertNotIn('4.8',cleaned['description'])
        self.assertEqual(diagnostic['discarded_passages']['rating_or_review_aggregate'],1)

    def test_conflicts_are_excluded_and_retained_for_recleaning(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder/'raw.jsonl'
            rows = [dict(parent_asin='P1',title='First title',description=[SYNOPSIS]),
                    dict(parent_asin='P1',title='Conflicting title',description=[SYNOPSIS]),
                    dict(parent_asin='P2',title='Distant Planet',description=[SYNOPSIS])]
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            output = folder/'clean'
            report = clean_domain(str(path), {'P1','P2','P3'}, '2023', 'books', output)
            self.assertEqual(report['accepted_items'], 1)
            self.assertEqual(report['missing_items'], 1)
            self.assertEqual(report['ambiguous_asins'], ['P1'])
            self.assertEqual(Path(report['source_snapshot']), path)
            self.assertEqual(len(path.read_text().splitlines()), 3)
            self.assertEqual(clean_domain(str(path), {'P1','P2','P3'}, '2023', 'books', output), report)
            original = path.read_text()
            path.write_text(original+'\n')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                clean_domain(str(path), {'P1','P2','P3'}, '2023', 'books', output)
            path.write_text(original)
            with (output/'books.jsonl').open('a') as handle:
                handle.write('{}\n')
            with self.assertRaises(ValueError):
                clean_domain(str(path), {'P1','P2','P3'}, '2023', 'books', output)

    def test_coverage_uses_source_histories_and_excludes_heldout_users_from_training(self):
        histories = np.zeros((4,40), dtype=np.int64)
        histories[:,0] = 1
        histories[0,20] = 2
        data = dict(source_item_ids=np.array(['B1','B2']), target_item_ids=np.array(['M1','M2']),
            source_train=np.array([[0,1,5],[0,2,1],[1,1,5],[2,1,5],[3,1,5]]),
            target_train=np.array([[1,3,4],[1,4,5]]), validation=np.array([[2,3,1]]), test=np.array([[3,4,5]]),
            shared_train_users=np.array([1]), validation_users=np.array([2]), test_users=np.array([3]), histories=histories)
        report = coverage_report(data, {'B1','B2'}, {'M1'}, minimum_books=2)
        self.assertEqual(report['populations']['source_only_training']['users'], 1)
        self.assertEqual(report['populations']['source_only_training']['eligible_at_minimum'], 1)
        self.assertEqual(report['populations']['source_only_training']['eligible_with_both_branches'], 1)
        self.assertEqual(report['warm_movies_with_content'], 1)
        data['validation'][:,2] = 5
        data['test'][:,2] = 1
        self.assertEqual(coverage_report(data, {'B1','B2'}, {'M1'}, 2), report)


if __name__ == '__main__':
    unittest.main()
