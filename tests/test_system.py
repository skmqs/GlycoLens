import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.preprocessing import FEATURES, load_data
from src.train import train, ROOT
from src.predict import load_models, predict_frame, validate_frame
from src.history import add, read, clear

class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=load_data(ROOT/'data/diabetes.csv')
        cls.models=load_models()
    def test_dataset_schema(self):
        self.assertEqual(self.df.shape,(768,9)); self.assertEqual(set(self.df.Outcome.unique()),{0,1})
    def test_saved_predictions_and_batch(self):
        x=self.df[FEATURES].iloc[[0,1]].copy()
        out=predict_frame(x,self.models)
        self.assertEqual(len(out),2)
        self.assertTrue(all(k+' prediction' in out for k in self.models))
    def test_validation(self):
        with self.assertRaises(ValueError): validate_frame(pd.DataFrame([{'Glucose':1}]))
        bad=pd.DataFrame([{**dict.fromkeys(FEATURES,1),'Age':121}])
        with self.assertRaises(ValueError): validate_frame(bad)
    def test_history_sqlite(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'history.sqlite3'
            add('test',1,.7,{'Age':33},path)
            rows=read(path); self.assertEqual(rows[0]['outcome'],1)
            clear(path); self.assertEqual(read(path),[])
    def test_training_artifacts_metrics(self):
        with tempfile.TemporaryDirectory() as t:
            result=train(ROOT/'data/diabetes.csv',Path(t))
            self.assertEqual(len(result['models']),3)
            for metrics in result['models'].values():
                for key in ('accuracy','precision','recall','f1','roc_auc'): self.assertGreaterEqual(metrics[key],0)
                self.assertEqual(sum(map(sum,metrics['confusion_matrix'])),154)

if __name__=='__main__': unittest.main()
