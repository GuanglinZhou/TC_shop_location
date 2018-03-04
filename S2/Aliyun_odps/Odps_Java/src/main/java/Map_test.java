import java.io.IOException;

import com.aliyun.odps.data.Record;
import com.aliyun.odps.mapred.MapperBase;

public class Map_test extends MapperBase {

    @Override
    public void setup(TaskContext context) throws IOException {

    }

    @Override
    public void map(long recordNum, Record record, TaskContext context) throws IOException {
        //TODO
    }

    @Override
    public void cleanup(TaskContext context) throws IOException {

    }

}