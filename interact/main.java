package interact;

public class main {
    public static void main(String[] args) {
        String content = String.format("name is: %s","MalGanis");
        System.out.println(content);
        System.out.println("What is that Thing?");
        frameWriter fw = new frameWriter();
        System.out.println(String.format("id is: %d",fw.getId()));
    }
}