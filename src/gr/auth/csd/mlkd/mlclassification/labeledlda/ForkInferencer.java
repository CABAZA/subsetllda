package gr.auth.csd.mlkd.mlclassification.labeledlda;
import gr.auth.csd.mlkd.mlclassification.labeledlda.models.Model;
import static java.util.Arrays.asList;
import java.util.concurrent.RecursiveAction;

/**
 *
 * @author Yannis Papanikolaou
 */
public class ForkInferencer extends RecursiveAction{

    static final int numberOfInstances = 2;
    protected Model newModel;
    private final int from, to;

    ForkInferencer(Model model, int from, int to) {
        newModel = model;
        this.from = from;
        this.to = to;
    }

    @Override
    protected void compute()
    {
        if(to - from <= numberOfInstances) {
            sampling(from, to);
         } else {
            int mid = from + (to - from) / 2;
            invokeAll(asList(new ForkInferencer(newModel, from, mid), new ForkInferencer(newModel, mid, to)));
         }
    }
    
    void sampling(int from, int to) {
        for(int i=from;i<to;i++) {
            newModel.update(i);
        }
    }
}
