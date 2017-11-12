import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by lerin on 10/11/2017.
 */
public class main {

    public static void main(String[] args) throws Exception{

        Scanner keyboard = new Scanner(System.in);
        System.out.println("Enter the route of the file");
        String input = keyboard.nextLine();
        String output = keyboard.nextLine();
        processData(input, output);


    }

    public static void processData(String inputfile, String outputfile) throws Exception{
        BufferedReader br = new BufferedReader(new FileReader(inputfile));
        String everything = "";

        try {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine().replaceAll("  ", " ");
            System.out.println(line);
            isMalicious(line);
            processLine(line, outputfile
            );

            while (line != null) {
                sb.append(line);
                sb.append(System.lineSeparator());
                line = br.readLine();
                if(line != null) line = line.replaceAll("  ", " ");
//                System.out.println(line + "\n");
                if(line != null) processLine(line, outputfile);
            }
            everything = sb.toString();
            br.close();
        } catch (IOException e) {

        }
    }

    public static String isMalicious(String line){
        String[] data = line.split(" ");
        String start = data[0];
        String end = data[data.length - 1];
        boolean virus = false;

        if(start.equals("-1") && end.equals("-1")) virus = true;

        if(virus == true) return "1";

        return "0";
    }

    public static void processLine(String line, String outputfile) throws Exception{
        String[] data = line.split(" ");
        List<String> result = new ArrayList<String>(Collections.nCopies(532, "0"));
//        String csvFile = "/Users/lerin/Desktop/test.csv";
        FileWriter writer = new FileWriter(outputfile, true);

        int i;
        for(i = 1; i < data.length - 1; i++){
            String[] element = data[i].split(":");
//            System.out.println(element[0]);
            result.set(Integer.parseInt(element[0]), "1");
        }
        result.set(0, isMalicious(line));
        System.out.println(result.toString());
        List<String> output = (result);
        CSVUtils.writeLine(writer, output);
        writer.flush();
        writer.close();

    }



}
