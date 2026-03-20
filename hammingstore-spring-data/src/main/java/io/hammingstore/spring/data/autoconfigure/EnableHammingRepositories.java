package io.hammingstore.spring.data.autoconfigure;

import io.hammingstore.spring.data.repository.HammingRepositoriesRegistrar;
import org.springframework.context.annotation.Import;

import java.lang.annotation.*;

@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Import(HammingRepositoriesRegistrar.class)
public @interface EnableHammingRepositories {

    String[] hasPackages() default {};
}
