import com.aliyun.odps.udf.ExecutionContext;
import com.aliyun.odps.udf.UDFException;
import com.aliyun.odps.udf.UDTF;
import com.aliyun.odps.udf.annotation.Resolve;

// TODO define input and output types, e.g. "string,string->string,bigint".
@Resolve({"string,string->string,string"})
public class test_bssid extends UDTF {

    @Override
    public void setup(ExecutionContext ctx) throws UDFException {

    }

    @Override
    public void process(Object[] args) throws UDFException {
        // TODO
        String mall_id = (String) args[0];
        String wifi_infos = (String) args[1];
        String bssid = "0";
        if (wifi_infos.indexOf(";") != -1) {
            String[] wifi_split = wifi_infos.split(";");
            //wifi_infos not connected
            for (int i = 0; i < wifi_split.length; i++) {
                bssid = wifi_split[i].split("\\|")[0].split("_")[1];
                forward(mall_id, bssid);

            }
        } else {
            bssid = wifi_infos.split("\\|")[0].split("_")[1];
            forward(mall_id, bssid);

        }
    }

    @Override
    public void close() throws UDFException {

    }

}