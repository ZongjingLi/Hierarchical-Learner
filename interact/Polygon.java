package interact;
import java.util.ArrayList;

public class Polygon {

    ArrayList<String> labels = new ArrayList<String>();
    
    public Polygon() {
        System.out.println("A Default Polygon is Created.");
        this.labels.add("Node1");
        this.labels.add("Node2");
        this.labels.add("Node3");
        int size = this.labels.size();
        for(int i = 0; i < size; i ++){
            System.out.println(this.labels.get(i));
        }
    }
}
