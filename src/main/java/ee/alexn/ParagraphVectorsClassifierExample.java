package ee.alexn;

import java.io.FileNotFoundException;
import java.util.List;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ParagraphVectorsClassifierExample {

	public static void main(String[] args) throws FileNotFoundException {

		ClassPathResource resource = new ClassPathResource("documents/labeled");

        // build a iterator for our dataset
        LabelAwareIterator iterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(resource.getFile())
                .build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
	        
		ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
				.learningRate(0.025)
				.minLearningRate(0.001)
				.batchSize(1000)
				.epochs(20)
				.iterate(iterator)
				.trainWordVectors(true)
				.tokenizerFactory(t)
				.build();
		
	   paragraphVectors.fit();
		
       ClassPathResource unlabeledResource = new ClassPathResource("documents/unlabeled");

       FileIterator unlabeledIterator = new FileIterator.Builder()
                .addSourceFolder(unlabeledResource.getFile())
                .build();
        
       MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), t);
       LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(), (InMemoryLookupTable<VocabWord>)  paragraphVectors.getLookupTable());

       while (unlabeledIterator.hasNextDocument()) {
           LabelledDocument document = unlabeledIterator.nextDocument();
           INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
           List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);
           System.out.println("Document '" + document.getLabel() + "' falls into the following categories: ");
           for (Pair<String, Double> score: scores) {
        	   System.out.println("        " + score.getFirst() + ": " + score.getSecond());
           }
       }
	}
	
}
