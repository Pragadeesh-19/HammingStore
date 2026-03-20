package io.hammingstore.spring.data.autoconfigure;

import io.hammingstore.client.HammingClient;
import io.hammingstore.spring.data.template.HammingTemplate;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.context.annotation.Bean;

import java.util.logging.Logger;

@AutoConfiguration
@ConditionalOnClass(HammingClient.class)
@ConditionalOnBean(HammingClient.class)
@ConditionalOnProperty(prefix = "hammingstore", name = "endpoint")
public class HammingDataAutoConfiguration {

    private static final Logger log =
            Logger.getLogger(HammingDataAutoConfiguration.class.getName());

    @Bean
    @ConditionalOnMissingBean(HammingTemplate.class)
    public HammingTemplate<?, ?> hammingTemplate(final HammingClient client,
                                                 final ApplicationEventPublisher publisher) {
        log.info("HammingStore: registering HammingTemplate");
        return new HammingTemplate<>(client, publisher);
    }
}
