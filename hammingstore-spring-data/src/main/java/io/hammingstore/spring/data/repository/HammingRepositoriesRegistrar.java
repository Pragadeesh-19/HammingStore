package io.hammingstore.spring.data.repository;

import io.hammingstore.spring.data.HammingOperations;
import io.hammingstore.spring.data.HammingRepository;
import io.hammingstore.spring.data.autoconfigure.EnableHammingRepositories;
import org.springframework.beans.factory.annotation.AnnotatedBeanDefinition;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.beans.factory.support.AbstractBeanDefinition;
import org.springframework.beans.factory.support.BeanDefinitionBuilder;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.context.annotation.ClassPathScanningCandidateComponentProvider;
import org.springframework.context.annotation.ImportBeanDefinitionRegistrar;
import org.springframework.core.type.AnnotationMetadata;
import org.springframework.core.type.filter.AnnotationTypeFilter;
import org.springframework.util.ClassUtils;

import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

public final class HammingRepositoriesRegistrar implements ImportBeanDefinitionRegistrar {

    private static final Logger log = Logger.getLogger(HammingRepositoriesRegistrar.class.getName());

    @Override
    public void registerBeanDefinitions(final AnnotationMetadata importingClassMetadata,
                                        final BeanDefinitionRegistry registry) {
        final String[] basePackages = resolveBasePackages(importingClassMetadata);
        final ClassPathScanningCandidateComponentProvider scanner = createScanner();

        for (final String pkg : basePackages) {
            log.info("HammingStore: scanning for @HammingRepository in package: " + pkg);
            final Set<BeanDefinition> candidates =
                    scanner.findCandidateComponents(pkg);

            for (final org.springframework.beans.factory.config.BeanDefinition candidate : candidates) {
                processCandidate(candidate.getBeanClassName(), registry);
            }
        }
    }

    private String[] resolveBasePackages(final AnnotationMetadata metadata) {
        final Map<String, Object> attrs = metadata
                .getAnnotationAttributes(EnableHammingRepositories.class.getName());

        if (attrs != null) {
            final String[] explicit = (String[]) attrs.get("basePackages");
            if (explicit != null && explicit.length > 0) {
                return explicit;
            }
        }
        return new String[]{ ClassUtils.getPackageName(metadata.getClassName()) };
    }

    private void processCandidate(final String className, final BeanDefinitionRegistry registry) {
        try {
            final Class<?> iface = ClassUtils.forName(className,
                    Thread.currentThread().getContextClassLoader());

            if (!iface.isInterface()) return;
            if (!iface.isAnnotationPresent(HammingRepository.class)) return;

            final Class<?> entityType = resolveEntityType(iface);
            if (entityType == null) {
                log.warning("HammingStore: cannot resolve entity type for "
                        + className + ". Ensure it extends HammingOperations<T, ID>. Skipping.");
                return;
            }

            registerFactoryBean(registry, iface, entityType);

        } catch (final ClassNotFoundException e) {
            log.warning("HammingStore: cannot load class " + className + ": " + e.getMessage());
        }
    }

    private void registerFactoryBean(final BeanDefinitionRegistry registry,
                                     final Class<?> repositoryInterface,
                                     final Class<?> entityType) {
        final String beanName = decapitalise(repositoryInterface.getSimpleName());

        if (registry.containsBeanDefinition(beanName)) {
            log.warning("HammingStore: bean '" + beanName
                    + "' already registered — skipping " + repositoryInterface.getName());
            return;
        }

        final AbstractBeanDefinition bd = BeanDefinitionBuilder
                .genericBeanDefinition(HammingRepositoryFactoryBean.class)
                .addConstructorArgValue(repositoryInterface)
                .addConstructorArgValue(entityType)
                .getBeanDefinition();

        registry.registerBeanDefinition(beanName, bd);
        log.info("HammingStore: registered repository bean '"
                + beanName + "' → " + repositoryInterface.getName()
                + " (entity=" + entityType.getSimpleName() + ")");
    }

    private ClassPathScanningCandidateComponentProvider createScanner() {
        final ClassPathScanningCandidateComponentProvider scanner =
                new ClassPathScanningCandidateComponentProvider(false) {
                    @Override
                    protected boolean isCandidateComponent(final AnnotatedBeanDefinition bmd) {
                        return bmd.getMetadata().isInterface();
                    }
                };
        scanner.addIncludeFilter(new AnnotationTypeFilter(HammingRepository.class));
        return scanner;
    }

    private Class<?> resolveEntityType(final Class<?> repositoryInterface) {
        for (final Type genericInterface : repositoryInterface.getGenericInterfaces()) {
            if (genericInterface instanceof ParameterizedType pt) {
                final String rawName = pt.getRawType().getTypeName();
                if (rawName.equals(HammingOperations.class.getName())) {
                    final Type[] args = pt.getActualTypeArguments();
                    if (args.length >= 1 && args[0] instanceof Class<?> cls) {
                        return cls;
                    }
                }
            }
        }
        return null;
    }

    private static String decapitalise(final String name) {
        if (name == null || name.isEmpty()) return name;
        return Character.toLowerCase(name.charAt(0)) + name.substring(1);
    }
}
