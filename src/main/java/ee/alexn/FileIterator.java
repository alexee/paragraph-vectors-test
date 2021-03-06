package ee.alexn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;

public class FileIterator implements LabelAwareIterator {
    protected List<File> files;
    protected AtomicInteger position = new AtomicInteger(0);
    protected LabelsSource labelsSource;

    /*
        Please keep this method protected, it's used in tests
     */
    protected FileIterator() {

    }

    protected FileIterator(List<File> files, LabelsSource source) {
        this.files = files;
        this.labelsSource = source;
    }

    @Override
    public boolean hasNextDocument() {
        return position.get() < files.size();
    }


    @Override
    public LabelledDocument nextDocument() {
        File fileToRead = files.get(position.getAndIncrement());
        try {
            LabelledDocument document = new LabelledDocument();
            BufferedReader reader = new BufferedReader(new FileReader(fileToRead));
            StringBuilder builder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) { 
            	builder.append(line);
            }

            document.setContent(builder.toString());
            document.setLabel(fileToRead.getName());

            return document;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void reset() {
        position.set(0);
    }

    @Override
    public LabelsSource getLabelsSource() {
        return labelsSource;
    }
	
	public static class Builder {
        protected List<File> foldersToScan = new ArrayList<>();

        public Builder() {

        }

        /**
         * Root folder for labels -> documents.
         * Each subfolder name will be presented as label, and contents of this folder will be represented as LabelledDocument, with label attached
         *
         * @param folder folder to be scanned for labels and files
         * @return
         */
        public Builder addSourceFolder(File folder) {
            foldersToScan.add(folder);
            return this;
        }

        public FileIterator build() {
            // search for all files in all folders provided
            List<File> fileList = new ArrayList<File>();
            List<String> labels = new ArrayList<String>();

            for (File file: foldersToScan) {
            	File[] files = file.listFiles();
            	for (File fileLabel : files) {
	            	if (fileLabel.isDirectory()) continue;
	
	                if (!labels.contains(fileLabel.getName())) labels.add(fileLabel.getName());
	
	                 fileList.add(fileLabel);
            	}
            }
            LabelsSource source = new LabelsSource(labels);
            FileIterator iterator = new FileIterator(fileList, source);

            return iterator;
        }
    }

}
